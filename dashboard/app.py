import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="VeritasBench", layout="wide")

st.title("VeritasBench — LLM Evaluation Dashboard")

def load_data():
    raw_scores = []
    aggregated_scores = {}
    agreement_data = {}
    
    raw_path = Path("results/raw_scores.json")
    if raw_path.exists():
        with open(raw_path, "r") as f:
            raw_scores = json.load(f)
            
    agg_path = Path("results/aggregated_scores.json")
    if agg_path.exists():
        with open(agg_path, "r") as f:
            aggregated_scores = json.load(f)
            
    agr_path = Path("results/agreement_report.json")
    if agr_path.exists():
        with open(agr_path, "r") as f:
            agreement_data = json.load(f)
            
    return raw_scores, aggregated_scores, agreement_data

raw_scores, aggregated_scores, agreement_data = load_data()

if not aggregated_scores:
    st.warning("No aggregated scores found. Please run the evaluation pipeline and aggregator.")
    st.stop()

# --- Radar Chart ---
st.header("1. Radar Chart")
radar_data = []
for model, data in aggregated_scores.items():
    dims = data["by_dimension"]
    for dim_name, stats in dims.items():
        radar_data.append(dict(Model=model, Dimension=dim_name, Score=stats["mean"]))
        
if radar_data:
    df_radar = pd.DataFrame(radar_data)
    fig = px.line_polar(df_radar, r='Score', theta='Dimension', color='Model', line_close=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# --- Overall Leaderboard ---
st.header("2. Overall Leaderboard")
leaderboard_data = []
for model, data in aggregated_scores.items():
    leaderboard_data.append({
        "Model": model,
        "Overall Score": data["overall"]["mean"],
        "Std Dev": data["overall"]["std"]
    })
df_lb = pd.DataFrame(leaderboard_data).sort_values(by="Overall Score", ascending=False)
st.dataframe(df_lb, use_container_width=True)

# --- Heatmap ---
st.header("3. Heatmap (Models vs Dimensions)")
heatmap_data = []
models = list(aggregated_scores.keys())
dimensions = list(aggregated_scores[models[0]]["by_dimension"].keys()) if models else []

for m in models:
    row = {"Model": m}
    for d in dimensions:
        row[d] = aggregated_scores[m]["by_dimension"][d]["mean"]
    heatmap_data.append(row)

if heatmap_data:
    df_hm = pd.DataFrame(heatmap_data).set_index("Model")
    fig_hm = px.imshow(df_hm, text_auto=True, color_continuous_scale="Viridis", template="plotly_dark")
    st.plotly_chart(fig_hm, use_container_width=True)

# --- Human vs Judge Agreement ---
st.header("4. Human vs Judge Agreement")
if agreement_data:
    col1, col2, col3 = st.columns(3)
    col1.metric("Comparisons", agreement_data.get("num_comparisons", 0))
    col2.metric("Raw Agreement", f"{agreement_data.get('raw_agreement_percent', 0):.1f}%")
    col3.metric("Cohen's Kappa", f"{agreement_data.get('cohens_kappa', 0):.3f}")
else:
    st.info("No agreement data found. Run human evaluation and agreement analysis.")

# --- Failure Analysis ---
st.header("5. Failure Analysis")
if raw_scores:
    failures = []
    for entry in raw_scores:
        if entry.get("failure_tags"):
            failures.append({
                "Model": entry["model_name"],
                "Prompt ID": entry["prompt_id"],
                "Category": entry["category"],
                "Tags": ", ".join(entry["failure_tags"])
            })
    if failures:
        df_failures = pd.DataFrame(failures)
        st.dataframe(df_failures, use_container_width=True)
        
        # Tag distribution
        tag_counts = pd.Series([t.strip() for tags in df_failures["Tags"] for t in tags.split(",")]).value_counts()
        fig_tags = px.bar(tag_counts, title="Failure Tag Frequency", template="plotly_dark")
        st.plotly_chart(fig_tags, use_container_width=True)

# --- Example Drill-down ---
st.header("6. Example Drill-down")
if raw_scores:
    df_raw = pd.DataFrame(raw_scores)
    sel_model = st.selectbox("Select Model", df_raw["model_name"].unique())
    sel_cat = st.selectbox("Select Category", df_raw["category"].unique())
    filtered = df_raw[(df_raw["model_name"] == sel_model) & (df_raw["category"] == sel_cat)]
    
    if not filtered.empty:
        sel_prompt_id = st.selectbox("Select Prompt ID", filtered["prompt_id"].unique())
        item = filtered[filtered["prompt_id"] == sel_prompt_id].iloc[0]
        
        st.subheader("Prompt")
        st.write(item["prompt"])
        st.subheader("Response")
        st.write(item["response"])
        st.subheader("Scores")
        st.json(item["dimension_scores"])
        st.subheader("Tags")
        st.write(item["failure_tags"] if item["failure_tags"] else "None")

# --- Cost Analytics ---
st.header("7. Cost Analytics")
if raw_scores:
    cost_data = []
    for entry in raw_scores:
        cost_data.append({
            "Model": entry["model_name"],
            "Tokens": entry.get("token_estimate", 0),
            "Latency (ms)": entry.get("latency_ms", 0)
        })
    df_cost = pd.DataFrame(cost_data).groupby("Model").sum().reset_index()
    st.dataframe(df_cost, use_container_width=True)

# --- Arena Rankings ---
st.header("8. Arena Rankings (ELO)")
st.info("ELO rankings are computed via the pairwise preference script and stored separately. Integration planned for next version if arena matches are generated.")
