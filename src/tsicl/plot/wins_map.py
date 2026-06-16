from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_win_rate_heatmap_simple(
    df_scores: pd.DataFrame,
    save_path: Path,
    top_k_y : int = 4,
    higher_is_better: bool = False
):
    """
    Generates a win rate heatmap comparing different methods,
    without confidence intervals.
    
    Arguments:
    - df_scores: Pandas DataFrame with N rows (datasets) and D columns (methods).
    - top_k_y: Number of top-performing methods to display on the Y-axis.
    - higher_is_better: True if a higher score is better (e.g., Accuracy), 
                        False if a lower score is better (e.g., RMSE, MAE).
    """
    methods = df_scores.columns
    N = len(df_scores)
    
    # 1. Calculate the win rate matrix (Y vs X)
    win_rates = pd.DataFrame(index=methods, columns=methods, dtype=float)
    
    for method_y in methods:
        for method_x in methods:
            if method_y == method_x:
                win_rates.loc[method_y, method_x] = 50.0
            else:
                if higher_is_better:
                    wins = (df_scores[method_y] > df_scores[method_x]).sum()
                else:
                    wins = (df_scores[method_y] < df_scores[method_x]).sum()
                
                ties = (df_scores[method_y] == df_scores[method_x]).sum()
                
                # Distribute ties 50/50
                win_pct = ((wins + 0.5 * ties) / N) * 100
                win_rates.loc[method_y, method_x] = win_pct

    # 2. Identify top methods (based on their global average win rate)
    avg_win_rates = win_rates.mean(axis=1)
    
    # Sort all methods from strongest to weakest
    sorted_methods = avg_win_rates.sort_values(ascending=False).index
    
    # Select Top K for Y-axis, keep all methods for X-axis
    top_methods_y = sorted_methods[:top_k_y]
    methods_x = sorted_methods 
    
    plot_data = win_rates.loc[top_methods_y, methods_x]

    # 3. Create annotation text (score only)
    annotations = pd.DataFrame(index=plot_data.index, columns=plot_data.columns, dtype=str)
    
    for y in plot_data.index:
        for x in plot_data.columns:
            val = plot_data.loc[y, x]
            # Display without decimals if it's a whole number
            if val.is_integer():
                annotations.loc[y, x] = f"{val:.0f}"
            else:
                annotations.loc[y, x] = f"{val:.1f}"

    # 4. Plot generation
    plt.figure(figsize=(10, 6)) 
    
    # Use PRGn (Purple-Green) palette centered at 50
    ax = sns.heatmap(plot_data, 
                     annot=annotations, 
                     fmt="", 
                     cmap="PRGn", 
                     vmin=0, vmax=100, center=50, 
                     cbar_kws={'label': 'Win Rate (%)'},
                     linewidths=0.5, linecolor='lightgray',
                     annot_kws={
                         "size": 16,          # Large font size for the scores
                        #  "weight": "bold",    # Bold scores inside cells
                         "va": "center"       # Vertical alignment
                     })
    
    # Aesthetics: Place X-axis labels at the top
    ax.xaxis.tick_top() 
    plt.xticks(rotation=90, fontsize=16, weight='bold')
    plt.yticks(rotation=0, fontsize=16, weight='bold')
    plt.xlabel("")
    plt.ylabel("")
    
    # Colorbar title and formatting
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Win Rate (%)', size=17, weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    data = {
        'TS-ICL': np.random.normal(30, 5, 100),
        'TabICLv2': np.random.normal(45, 8, 100),
        'TabPFNv2.5': np.random.normal(52, 10, 100),
        'MoTM': np.random.normal(60, 12, 100),
        'Linear': np.random.normal(70, 15, 100),
        'Seasonnal': np.random.normal(70, 15, 100),
        'LOCF': np.random.normal(80, 20, 100),
    }
    
    df_fictif = pd.DataFrame(data)
    plot_win_rate_heatmap_simple(df_fictif, save_path=Path('./test_wins.pdf'), top_k_y=4, higher_is_better=False)