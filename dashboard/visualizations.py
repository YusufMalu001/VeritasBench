import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any

def create_radar_chart(agg_scores: Dict[str, Any], models: list) -> go.Figure:
    fig = go.Figure()
    
    categories = ['instruction_following', 'factuality', 'format_adherence', 'refusal_calibration', 'verbosity']
    
    for model in models:
        if model in agg_scores:
            by_cat = agg_scores[model].get("by_category", {})
            scores = [by_cat.get(cat, {}).get("mean", 0.0) for cat in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=[cat.replace("_", " ").title() for cat in categories] + [categories[0].replace("_", " ").title()],
                fill='toself',
                name=model
            ))
            
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_easy_vs_hard_chart(results_df: pd.DataFrame, model_name: str) -> go.Figure:
    if results_df.empty or "tier" not in results_df.columns:
        return go.Figure()
        
    model_df = results_df[results_df["model_name"] == model_name]
    
    if model_df.empty:
        return go.Figure()
        
    agg_df = model_df.groupby(["category", "tier"])["score"].mean().reset_index()
    agg_df["category"] = agg_df["category"].str.replace("_", " ").str.title()
    
    fig = px.bar(
        agg_df, 
        x="category", 
        y="score", 
        color="tier", 
        barmode="group",
        title=f"Easy vs Hard Tier Performance: {model_name}",
        template="plotly_dark",
        color_discrete_map={"easy": "#00CC96", "hard": "#EF553B"}
    )
    
    fig.update_layout(yaxis_range=[0, 1])
    return fig

def create_failure_analysis_chart(results_df: pd.DataFrame, model_name: str) -> go.Figure:
    if results_df.empty:
        return go.Figure()
        
    model_df = results_df[results_df["model_name"] == model_name]
    
    tags = []
    for tag_str in model_df["failure_tags"].dropna():
        if tag_str:
            tags.extend([t.strip() for t in tag_str.split(",") if t.strip()])
            
    if not tags:
        fig = go.Figure()
        fig.update_layout(title="No Failures Detected", template="plotly_dark")
        return fig
        
    tag_counts = pd.Series(tags).value_counts().reset_index()
    tag_counts.columns = ["Failure Tag", "Count"]
    
    fig = px.bar(
        tag_counts, 
        x="Count", 
        y="Failure Tag", 
        orientation="h",
        title=f"Top Failure Modes: {model_name}",
        template="plotly_dark",
        color="Count",
        color_continuous_scale="Reds"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    
    return fig
