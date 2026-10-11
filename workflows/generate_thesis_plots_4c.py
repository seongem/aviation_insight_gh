import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import theme from palette manager logic
from palette_manager import theme

theme.apply_publication_style()

# Let's create a 2x2 grid script to test the feature mask panels with mock data across 6 models
models = ['OmniAnomaly', 'PatchTST', 'TS2Vec', 'LSTM-VAE', 'Anomaly Transformer', 'ConvAutoencoder']
masks = ['enabled_all', 'enabled_baseline', 'enabled_baseline_acc', 'enabled_baseline_piml']
fractions = [1.0, 0.75, 0.50, 0.25]

# Generate structured mock data
mock_data = []
np.random.seed(42)
for m_idx, m in enumerate(models):
    for mask in masks:
        base_val = 0.25 + (m_idx * 0.01)
        for frac in fractions:
            val = base_val + np.random.normal(0, 0.015)
            mock_data.append({'Model': m, 'Feature Mask': mask, 'Data Fraction': frac, 'PR-AUC': val})

df_mock = pd.DataFrame(mock_data)

# Map 6 models to the Okabe-Ito palette colors
model_colors = {models[i]: theme.OKABE_ITO[i % len(theme.OKABE_ITO)] for i in range(len(models))}
mask_titles = {
    'enabled_all': 'Feature Mask: All (Baseline + Acc + PIML)',
    'enabled_baseline': 'Feature Mask: Baseline Only',
    'enabled_baseline_acc': 'Feature Mask: Baseline + Accelerations',
    'enabled_baseline_piml': 'Feature Mask: Baseline + PIML Residuals'
}

fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=theme.DPI, sharex=True, sharey=True)
axes_flat = axes.flatten()

for idx, mask in enumerate(masks):
    ax = axes_flat[idx]
    sub_mask_df = df_mock[df_mock['Feature Mask'] == mask]
    
    for m in models:
        sub = sub_mask_df[sub_mask_df['Model'] == m].sort_values('Data Fraction', ascending=False)
        ax.plot(
            sub['Data Fraction'],
            sub['PR-AUC'],
            color=model_colors[m],
            marker='o',
            linewidth=1.5,
            markersize=5,
            label=m
        )
    
    ax.set_title(mask_titles[mask], fontsize=10, fontweight='bold')
    ax.set_xlabel("Training Data Fraction", fontsize=9)
    ax.set_ylabel("PR-AUC Score", fontsize=9)
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.5)

# Place a single shared legend outside or at the bottom
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=8)

plt.suptitle("Multi-Model Data Scarcity Performance across Feature Configurations", fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()

os.makedirs("outputs/evaluation_plots", exist_ok=True)
save_path = "outputs/evaluation_plots/scarcity_2x2_grid_test.png"
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Successfully generated 2x2 grid test plot at {save_path}")