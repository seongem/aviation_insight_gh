import os
import sys
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Project root injection
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate Thesis Evaluation Plots")
parser.add_argument(
    "--config", 
    type=str, 
    required=True, 
    help="Path or name of the experiment YAML file (e.g., experiments_features.yaml)"
)
parser.add_argument(
    "--environment", 
    type=str, 
    required=True, 
    help="Environment profile name (e.g., local_dev, local_prod, colab)"
)
args = parser.parse_args()

# 1. Load Environment Configuration from environment.yaml
env_file_path = Path(project_root) / "configs" / "environment.yaml"
if not env_file_path.exists():
    raise FileNotFoundError(f"Could not find environment config at: {env_file_path}")

with open(env_file_path, "r") as f:
    env_config = yaml.safe_load(f)

environments = env_config.get("environments", {})
if args.environment not in environments:
    raise ValueError(f"Environment '{args.environment}' not found in {env_file_path}. Available: {list(environments.keys())}")

env_paths = environments[args.environment]
OUTPUT_DIR = Path(env_paths["OUTPUT_DIR"])

# 2. Load Experiment Configuration to extract the experiment_tag
# Handle paths relative to project root or configs directory if needed
config_path = Path(args.config)
if not config_path.is_absolute():
    # Try looking in configs/ directory first if just filename provided
    if (Path(project_root) / "configs" / config_path).exists():
        config_path = Path(project_root) / "configs" / config_path
    else:
        config_path = Path(project_root) / config_path

if not config_path.exists():
    raise FileNotFoundError(f"Could not find config file at: {config_path}")

with open(config_path, "r") as f:
    exp_config = yaml.safe_load(f)

experiment_tag = exp_config.get("experiment_tag", "default_experiment")

# Import theme from palette manager logic
from src.utils.palette_manager import theme
theme.apply_publication_style()

# 3. Locate CSV file using environment OUTPUT_DIR and experiment_tag folder name
csv_path = OUTPUT_DIR / "evaluation_plots" / experiment_tag / "benchmark_summary_all_masks.csv"

if not csv_path.exists():
    # Fallback check if evaluation_results.csv is placed directly inside OUTPUT_DIR
    csv_path_fallback = OUTPUT_DIR / "benchmark_summary_all_masks.csv"
    if csv_path_fallback.exists():
        csv_path = csv_path_fallback
    else:
        raise FileNotFoundError(f"Could not find the evaluation CSV file at: {csv_path} or {csv_path_fallback}")

# Load actual data from CSV
df = pd.read_csv(csv_path)

# Dynamically extract unique models and feature masks from the dataset
models = df['Model'].unique().tolist()
masks = df['Feature Mask'].unique().tolist()

# Map models to the Okabe-Ito palette colors
model_colors = {models[i]: theme.OKABE_ITO[i % len(theme.OKABE_ITO)] for i in range(len(models))}

# Optional custom titles for known feature masks
mask_titles = {
    'enabled_all': 'Feature Mask: All (Baseline + Acc + PIML)',
    'enabled_baseline': 'Feature Mask: Baseline Only',
    'enabled_baseline_acc': 'Feature Mask: Baseline + Accelerations',
    'enabled_baseline_piml': 'Feature Mask: Baseline + PIML Residuals'
}

def get_mask_title(mask_name):
    return mask_titles.get(mask_name, f"Feature Mask: {mask_name}")

# Create a 2x2 grid (or adjust subplot grid size dynamically if needed)
fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=theme.DPI, sharex=True, sharey=True)
axes_flat = axes.flatten()

for idx, mask in enumerate(masks):
    if idx >= len(axes_flat):
        break  # Safety check in case there are more than 4 masks
        
    ax = axes_flat[idx]
    sub_mask_df = df[df['Feature Mask'] == mask]
    
    for m in models:
        sub = sub_mask_df[sub_mask_df['Model'] == m].sort_values('Data Fraction', ascending=False)
        if sub.empty:
            continue
        ax.plot(
            sub['Data Fraction'],
            sub['PR-AUC'],
            color=model_colors[m],
            marker='o',
            linewidth=1.5,
            markersize=5,
            label=m
        )
    
    ax.set_title(get_mask_title(mask), fontsize=10, fontweight='bold')
    ax.set_xlabel("Training Data Fraction", fontsize=9)
    ax.set_ylabel("PR-AUC Score", fontsize=9)
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.5)

# Place a single shared legend at the bottom
handles, labels = axes_flat[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=8)

plt.suptitle(f"Multi-Model Data Scarcity Performance ({experiment_tag})", fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save final plot into the respective environment output directory + experiment folder
plot_output_dir = OUTPUT_DIR / "evaluation_plots" / experiment_tag
os.makedirs(plot_output_dir, exist_ok=True)
save_path = plot_output_dir / "scarcity_2x2_grid.svg"
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Successfully generated 2x2 grid plot for environment '{args.environment}' and config '{args.config}' at {save_path}")