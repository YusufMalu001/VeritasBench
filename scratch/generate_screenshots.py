import sys
import os
from pathlib import Path
import plotly.io as pio

# Add parent to path to import dashboard modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.data_loader import load_dashboard_data
from dashboard.visualizations import create_radar_chart, create_easy_vs_hard_chart, create_failure_analysis_chart

def main():
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    results_df, agg_scores, difficulty, eval_errors, prompts_df, errors_df = load_dashboard_data()
    
    if results_df.empty:
        print("Error: No data available to generate screenshots.")
        return
        
    models = results_df["model_name"].unique().tolist()
    if not models:
        print("No models found in data.")
        return
        
    primary_model = models[0]
    
    print("Generating radar chart...")
    radar_fig = create_radar_chart(agg_scores, models)
    radar_fig.write_image(assets_dir / "radar_chart.png", width=800, height=600, scale=2)
    
    print("Generating benchmark comparison chart...")
    if "tier" in results_df.columns:
        benchmark_fig = create_easy_vs_hard_chart(results_df, primary_model)
        benchmark_fig.write_image(assets_dir / "benchmark_comparison.png", width=800, height=500, scale=2)
    else:
        print("Skipping benchmark comparison (no tier data)")
        
    print("Generating failure analysis chart...")
    failure_fig = create_failure_analysis_chart(results_df, primary_model)
    failure_fig.write_image(assets_dir / "failure_analysis.png", width=800, height=500, scale=2)
    
    # We can't trivially screenshot the whole Streamlit overview without playwright,
    # so we'll just log that the asset is meant to be a full page snip and we generate the sub-components.
    print(f"Screenshots generated successfully in {assets_dir.absolute()}")
    
if __name__ == "__main__":
    main()
