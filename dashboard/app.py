import streamlit as st
import pandas as pd
from data_loader import load_dashboard_data
from visualizations import create_radar_chart, create_easy_vs_hard_chart, create_failure_analysis_chart

# Configure Page
st.set_page_config(
    page_title="VeritasBench Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
@st.cache_data
def get_data():
    return load_dashboard_data()

results_df, agg_scores, difficulty, eval_errors, prompts_df, errors_df = get_data()

if results_df.empty:
    st.error("No benchmark data found. Please run the evaluation pipeline first.")
    st.stop()

# Sidebar Setup
st.sidebar.title("VeritasBench Options")
available_models = results_df["model_name"].unique().tolist()
selected_model = st.sidebar.selectbox("Select Model", available_models)

# App Title
st.title("VeritasBench Evaluation Dashboard")

# ==================================================
# SECTION 1: OVERVIEW
# ==================================================
st.header("1. Overview")

# Calculate metrics for selected model
model_results = results_df[results_df["model_name"] == selected_model]
overall_score = 0.0
if selected_model in agg_scores:
    overall_score = agg_scores[selected_model].get("overall", {}).get("mean", 0.0)

total_prompts = len(model_results)
avg_latency = model_results["latency_ms"].mean() if not model_results.empty else 0
eval_fails = model_results["evaluation_failed"].sum()

# Assume we take benchmark version from the first prompt
version = prompts_df["benchmark_version"].iloc[0] if not prompts_df.empty else "v1.0"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Model", selected_model.split("/")[-1])
col2.metric("Benchmark Version", version)
col3.metric("Overall Score", f"{overall_score:.2%}")
col4.metric("Avg Latency", f"{avg_latency/1000:.2f}s")
col5.metric("Evaluation Failures", int(eval_fails))

st.markdown("---")

# ==================================================
# SECTION 2: RADAR CHART
# ==================================================
st.header("2. Model Comparison (Radar Chart)")
if len(available_models) > 1:
    radar_models = st.multiselect("Select models to compare", available_models, default=available_models)
else:
    radar_models = [selected_model]

radar_fig = create_radar_chart(agg_scores, radar_models)
st.plotly_chart(radar_fig, use_container_width=True)

st.markdown("---")

# ==================================================
# SECTION 3: EASY VS HARD BENCHMARK
# ==================================================
st.header("3. Easy vs Hard Tier Performance")
if "tier" in results_df.columns:
    easy_hard_fig = create_easy_vs_hard_chart(results_df, selected_model)
    st.plotly_chart(easy_hard_fig, use_container_width=True)
else:
    st.info("Tier data not available. Ensure V2.1 benchmark structure is used.")

st.markdown("---")

# ==================================================
# SECTION 4: FAILURE ANALYSIS
# ==================================================
st.header("4. Failure Analysis")
failure_fig = create_failure_analysis_chart(results_df, selected_model)
st.plotly_chart(failure_fig, use_container_width=True)

st.markdown("---")

# ==================================================
# SECTION 5: BENCHMARK DIFFICULTY
# ==================================================
st.header("5. Benchmark Difficulty Report")
diff_data = difficulty.get("prompt_classification", {})

col1, col2, col3 = st.columns(3)
col1.metric("Easy Prompts", len(diff_data.get("Easy", [])))
col2.metric("Medium Prompts", len(diff_data.get("Medium", [])))
col3.metric("Hard Prompts", len(diff_data.get("Hard", [])))

sat_dims = difficulty.get("saturated_dimensions", [])
if sat_dims:
    st.warning(f"⚠️ **Saturation Detected**: The following dimensions scored 100% with 0 variance: {', '.join(sat_dims)}. Consider increasing prompt difficulty.")

st.markdown("---")

# ==================================================
# SECTION 6: EVALUATION INTEGRITY PANEL
# ==================================================
st.header("6. Evaluation Integrity Panel")

col1, col2, col3 = st.columns(3)

config_errs_count = len(errors_df)
if config_errs_count > 0:
    col1.error(f"Config Errors: {config_errs_count}")
else:
    col1.success("Config Errors: 0")

if eval_fails > 0:
    col2.warning(f"Evaluation Failures: {int(eval_fails)}")
else:
    col2.success("Evaluation Failures: 0")

if sat_dims:
    col3.warning(f"Zero Variance Dimensions: {len(sat_dims)}")
else:
    col3.success("Zero Variance Dimensions: 0")

st.markdown("---")

# ==================================================
# SECTION 7: PROMPT DRILL-DOWN
# ==================================================
st.header("7. Prompt Drill-Down")

st.dataframe(
    model_results[["prompt_id", "category", "tier", "score", "failure_tags"]],
    use_container_width=True,
    hide_index=True
)

st.subheader("Detail View")
drill_id = st.selectbox("Select Prompt ID for details", model_results["prompt_id"].tolist())

if drill_id:
    drill_data = model_results[model_results["prompt_id"] == drill_id].iloc[0]
    
    st.markdown("**Prompt:**")
    st.info(drill_data["prompt"])
    
    st.markdown("**Response:**")
    st.success(drill_data["response"] if drill_data["response"] else "No response generated.")
    
    st.markdown("**Explanation:**")
    st.code(drill_data["explanation"])

st.markdown("---")

# ==================================================
# SECTION 8: MODEL COMPARISON
# ==================================================
st.header("8. Model Comparison (Leaderboard)")

leaderboard_data = []
for m in available_models:
    if m in agg_scores:
        overall = agg_scores[m].get("overall", {}).get("mean", 0.0)
        leaderboard_data.append({"Model": m, "Overall Score": f"{overall:.2%}"})

leaderboard_df = pd.DataFrame(leaderboard_data)
st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ==================================================
# SECTION 9: COST ANALYTICS
# ==================================================
st.header("9. Cost Analytics")

avg_tokens = model_results["token_estimate"].mean() if not model_results.empty else 0
# Approx cost heuristic: $0.50 / 1M tokens
estimated_cost = (avg_tokens * total_prompts) / 1_000_000 * 0.50

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Latency", f"{avg_latency/1000:.2f}s")
col2.metric("Average Tokens", f"{avg_tokens:.0f}")
col3.metric("Estimated Cost", f"${estimated_cost:.4f}")
col4.metric("Cache Hit Rate", "0.0%") # Placeholder for actual cache stat if added

st.markdown("---")

# ==================================================
# SECTION 10: BIAS ANALYSIS
# ==================================================
st.header("10. Bias Analysis")
import json
from pathlib import Path

bias_file = Path("results/bias_report.json")
if bias_file.exists():
    with open(bias_file, "r", encoding="utf-8") as f:
        bias_data = json.load(f)
        
    col1, col2, col3, col4 = st.columns(4)
    
    lb_corr = bias_data.get("length_bias", {}).get("correlation", 0.0)
    pos_cons = bias_data.get("position_bias", {}).get("consistency_rate", 1.0)
    hack = bias_data.get("hackability_score", 0.0)
    cal_err = bias_data.get("calibration_error", {}).get("calibration_error", 0.0)
    
    def get_color(val, threshold1, threshold2, reverse=False):
        if reverse:
            if val > threshold2: return "green"
            if val > threshold1: return "orange"
            return "red"
        else:
            if val < threshold1: return "green"
            if val < threshold2: return "orange"
            return "red"
            
    lb_color = get_color(abs(lb_corr), 0.2, 0.4)
    pos_color = get_color(pos_cons, 0.6, 0.8, reverse=True)
    hack_color = get_color(hack, 0.2, 0.4)
    cal_color = get_color(cal_err, 0.2, 0.4)
    
    col1.markdown(f"**Length Bias (r)**<br><h3 style='color:{lb_color}'>{abs(lb_corr):.3f}</h3>", unsafe_allow_html=True)
    col2.markdown(f"**Position Consistency**<br><h3 style='color:{pos_color}'>{pos_cons:.1%}</h3>", unsafe_allow_html=True)
    col3.markdown(f"**Hackability Score**<br><h3 style='color:{hack_color}'>{hack:.3f}</h3>", unsafe_allow_html=True)
    col4.markdown(f"**Calibration Error**<br><h3 style='color:{cal_color}'>{cal_err:.3f}</h3>", unsafe_allow_html=True)
    
    with st.expander("What do these metrics mean?"):
        st.write("- **Length Bias**: Pearson correlation between response length and judge score. >0.3 indicates verbosity bias.")
        st.write("- **Position Consistency**: How often the judge gives the same verdict when A/B order is swapped. <85% indicates positional bias.")
        st.write("- **Hackability Score**: Percentage of high-scoring responses that are also excessively long.")
        st.write("- **Calibration Error**: Average of false-refusal rate and false-compliance rate.")
else:
    st.info("Run `python -m analysis.bias_detector` to generate bias analytics.")

st.markdown("---")

# ==================================================
# SECTION 11: HEAD-TO-HEAD DEEP DIVE
# ==================================================
st.header("11. Head-to-Head Deep Dive")
disagree_file = Path("results/disagreement_analysis.json")
if disagree_file.exists():
    with open(disagree_file, "r", encoding="utf-8") as f:
        disagree_data = json.load(f)
        
    top_disagreements = disagree_data.get("top_disagreements", [])
    if top_disagreements:
        df_disagree = pd.DataFrame(top_disagreements)
        
        dim_filter = st.selectbox("Filter by Dimension", ["All"] + df_disagree["category"].unique().tolist())
        if dim_filter != "All":
            df_disagree = df_disagree[df_disagree["category"] == dim_filter]
            
        st.dataframe(
            df_disagree[["prompt_id", "category", "winner", "delta"]].style.highlight_max(subset=['delta'], color='rgba(255, 75, 75, 0.3)'),
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Run `python -m analysis.comparative_analysis` to generate head-to-head metrics.")

st.markdown("---")

# ==================================================
# SECTION 12: BENCHMARK HEALTH
# ==================================================
st.header("12. Benchmark Health")
diff_file = Path("results/difficulty_analysis.json")
if diff_file.exists():
    with open(diff_file, "r", encoding="utf-8") as f:
        diff_data = json.load(f)
        
    mislabeled = diff_data.get("mislabeled_tiers", [])
    if mislabeled:
        st.warning(f"⚠️ {len(mislabeled)} prompts may be miscategorized across Easy/Hard tiers.")
        st.dataframe(pd.DataFrame(mislabeled), use_container_width=True, hide_index=True)
    else:
        st.success("No mislabeled tiers detected.")
        
    # Histogram
    item_diffs = diff_data.get("item_difficulties", {})
    if item_diffs:
        diff_df = pd.DataFrame([{"prompt_id": k, "difficulty": v} for k,v in item_diffs.items()])
        st.subheader("Difficulty Distribution")
        import plotly.express as px
        fig = px.histogram(diff_df, x="difficulty", nbins=20, title="Empirical Difficulty of Prompts", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Run `python -m analysis.difficulty_calibrator` to assess benchmark health.")

st.markdown("---")
st.caption("VeritasBench Evaluation Dashboard • Powered by Streamlit")
