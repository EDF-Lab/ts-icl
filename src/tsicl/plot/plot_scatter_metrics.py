import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_mase_vs_crps(df_mase, df_crps, dict_categories):
    """
    Generates a 2D scatter plot of mean MASE vs CRPS scores with bold aesthetics.
    """
    
    # 1. Calculate mean scores
    mean_mase = df_mase.mean()
    mean_crps = df_crps.mean()
    
    # 2. Prepare DataFrame
    df_plot = pd.DataFrame({
        'MASE': mean_mase,
        'CRPS': mean_crps,
        'Method': mean_mase.index
    })
    df_plot['Category'] = df_plot['Method'].map(dict_categories).fillna('Unknown')


    # 3. Visual configuration - BOLD GRID AND AXES
    sns.set_theme(style="whitegrid", rc={
        "grid.color": "#9B9191",      # Darker grey for the grid
        "grid.linestyle": "-",
        "grid.linewidth": 1.2         # Thicker grid lines
    })
    
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = {"Foundation models": "#d95f02", "Local methods": "#1b9e77", "Unknown": "gray"}
    
    # 4. Plot points
    sns.scatterplot(
        data=df_plot, 
        x='MASE', 
        y='CRPS', 
        hue='Category',
        s=250,                        
        palette=palette,
        edgecolor='white',
        linewidth=2,                  
        ax=ax,
        zorder=3
    )
    
    # 5. Bold labels for each method
    for i, row in df_plot.iterrows():
        ax.annotate(
            row['Method'], 
            (row['MASE'], row['CRPS']),
            xytext=(12, 0),
            textcoords='offset points',
            fontsize=15,
            fontweight='bold',
            va='center', 
            color="#222222",
            zorder=4
        )
        
    # 6. Formatting axes and bold ticks
    ax.set_xlabel('MASE', fontsize=16, fontweight='bold', color="#222222")
    ax.set_ylabel('CRPS', fontsize=16, fontweight='bold', color="#222222")
    
    # Make the tick labels (numbers) bold
    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Make the spine (axis lines) thicker and darker
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("#676262")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#514D4D")
    
    # Legend below plot
    ax.legend(
        prop={'size': 15, 'weight': 'bold'},
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
        edgecolor="#746D6D"
    )
    
    plt.tight_layout(rect=(0, 0.18, 1, 1)) 
    plt.show()

# ==========================================
# EXAMPLE USAGE WITH DUMMY DATA
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    
    # Methods list
    methods = ['Chronos-2', 'TimesFM-2.5', 'Moirai2', 'TS-ICL', 'Prophet', 'ETS', 'Linear']
    
    # Generate dummy data
    df_dummy_mase = pd.DataFrame({m: np.random.normal(loc=np.random.uniform(0.4, 0.6), scale=0.05, size=100) for m in methods})
    df_dummy_crps = pd.DataFrame({m: np.random.normal(loc=np.random.uniform(0.4, 0.6), scale=0.02, size=100) for m in methods})
    
    # Mapping methods to categories
    categories = {
        'Chronos-2': 'Foundation models',
        'TimesFM-2.5': 'Foundation models',
        'Moirai2': 'Foundation models',
        'TS-ICL': 'Foundation models',
        'Prophet': 'Local methods',
        'ETS': 'Local methods',
        'Linear': 'Local methods'
    }
    
    plot_mase_vs_crps(df_dummy_mase, df_dummy_crps, categories)