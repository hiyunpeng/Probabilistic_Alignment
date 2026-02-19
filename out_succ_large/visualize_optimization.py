#!/usr/bin/env python3
"""
Advanced Box Plot Generator for Optimization Algorithm Benchmarks
Creates multiple types of box plots to analyze algorithm performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(filename):
    """Load and preprocess the benchmark data"""
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)

    # Clean up algorithm names
    df['algo_clean'] = df['algo_variant'].apply(
        lambda x: x.split('(')[0] if '(' in x else x
    )

    # Extract problem size if available
    try:
        df['problem_size'] = df['instance_id'].str.extract('n=(\d+)').astype(float)
    except:
        df['problem_size'] = np.nan

    print(f"Loaded {len(df)} configurations")
    return df


def create_output_dir():
    """Create output directory for plots"""
    output_dir = 'boxplot_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ============================================================================
# BOX PLOT TYPE 1: Standard Box Plot - Success Rates by Algorithm
# ============================================================================
def plot_success_rate_boxplot(df, output_dir):
    """Create box plot of success rates for each algorithm"""

    plt.figure(figsize=(14, 8))

    # Select top 20 algorithms by mean success rate
    top_algos = df.groupby('algo_clean')['succ_rate'].mean().nlargest(20).index
    df_top = df[df['algo_clean'].isin(top_algos)]

    # Create box plot
    sns.boxplot(data=df_top, x='algo_clean', y='succ_rate',
                palette='viridis', width=0.7, linewidth=1.5)

    # Add individual points (strip plot) for better visualization
    sns.stripplot(data=df_top, x='algo_clean', y='succ_rate',
                  color='black', alpha=0.3, size=3, jitter=True)

    plt.title('Distribution of Success Rates by Algorithm\n(Box Plot with Individual Points)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '01_success_rate_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 2: Grouped Box Plot - Algorithms by Domain
# ============================================================================
def plot_domain_grouped_boxplot(df, output_dir):
    """Create grouped box plot showing algorithm performance across domains"""

    plt.figure(figsize=(18, 8))

    # Select representative algorithms
    rep_algos = ['HC', 'SA', 'GA', 'DE', 'PSO_STD', 'PSO_RING', 'TABU', 'UMDA', 'ES_ML']
    df_rep = df[df['algo_clean'].isin(rep_algos)].copy()

    # Create grouped box plot
    sns.boxplot(data=df_rep, x='domain', y='succ_rate', hue='algo_clean',
                palette='tab10', width=0.7, linewidth=1)

    plt.title('Algorithm Performance Across Different Problem Domains\n(Grouped Box Plot)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Problem Domain', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '02_domain_grouped_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 3: Violin Plot (Enhanced Box Plot)
# ============================================================================
def plot_violin_distribution(df, output_dir):
    """Create violin plots showing full distribution with kernel density"""

    plt.figure(figsize=(14, 8))

    # Select top algorithms
    top_algos = df.groupby('algo_clean')['succ_rate'].mean().nlargest(15).index
    df_top = df[df['algo_clean'].isin(top_algos)]

    # Create violin plot with quartiles shown
    sns.violinplot(data=df_top, x='algo_clean', y='succ_rate',
                   palette='muted', cut=0, inner='quartile')

    plt.title('Distribution Shape of Success Rates\n(Violin Plot with Quartiles)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '03_violin_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 4: Horizontal Box Plot with Mean Points
# ============================================================================
def plot_horizontal_boxplot_with_means(df, output_dir):
    """Create horizontal box plots with mean markers"""

    plt.figure(figsize=(10, 12))

    # Select top algorithms
    top_algos = df.groupby('algo_clean')['succ_rate'].mean().nlargest(15).index
    df_top = df[df['algo_clean'].isin(top_algos)]

    # Create horizontal box plot
    ax = sns.boxplot(data=df_top, y='algo_clean', x='succ_rate',
                     palette='coolwarm', orient='h', width=0.7)

    # Calculate and plot means
    means = df_top.groupby('algo_clean')['succ_rate'].mean()
    for i, (algo, mean_val) in enumerate(means.items()):
        plt.plot(mean_val, i, 'rD', markersize=8, label='Mean' if i == 0 else '')

    plt.title('Algorithm Performance Comparison\n(Horizontal Box Plot with Mean Markers)',
              fontsize=16, fontweight='bold')
    plt.ylabel('Algorithm', fontsize=12)
    plt.xlabel('Success Rate', fontsize=12)
    plt.legend(['Mean'], loc='best')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '04_horizontal_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 5: Notched Box Plot (for significance testing)
# ============================================================================
def plot_notched_boxplot(df, output_dir):
    """Create notched box plots - notches show confidence intervals around median"""

    plt.figure(figsize=(14, 8))

    # Select top algorithms for readability
    top_algos = df.groupby('algo_clean')['succ_rate'].mean().nlargest(12).index
    df_top = df[df['algo_clean'].isin(top_algos)]

    # Create notched box plot
    sns.boxplot(data=df_top, x='algo_clean', y='succ_rate',
                palette='Set3', notch=True, width=0.7, linewidth=1.5)

    plt.title('Notched Box Plot - Confidence Intervals Around Medians\n(Overlapping notches suggest similar medians)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '05_notched_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 6: Performance vs Problem Size (Faceted Box Plots)
# ============================================================================
def plot_size_faceted_boxplots(df, output_dir):
    """Create faceted box plots showing performance across problem sizes"""

    # Create size categories
    df['size_category'] = pd.cut(df['problem_size'],
                                 bins=[0, 50, 100, 150, float('inf')],
                                 labels=['Small (≤50)', 'Medium (51-100)',
                                         'Large (101-150)', 'Extra Large (>150)'])

    # Filter out NaN
    df_size = df[df['size_category'].notna()]

    if len(df_size) == 0:
        print("No problem size data available for faceted plots")
        return

    # Select representative algorithms
    rep_algos = ['HC', 'SA', 'GA', 'DE', 'PSO_STD']
    df_rep = df_size[df_size['algo_clean'].isin(rep_algos)]

    # Create faceted plot
    g = sns.FacetGrid(df_rep, col='size_category', col_wrap=2,
                      height=5, aspect=1.2, sharey=True)
    g.map_dataframe(sns.boxplot, x='algo_clean', y='succ_rate', palette='Set2')
    g.set_axis_labels('Algorithm', 'Success Rate')
    g.set_titles(col_template='{col_name}')
    g.add_legend()
    g.tight_layout()

    plt.suptitle('Algorithm Performance Across Problem Sizes\n(Faceted Box Plots)',
                 y=1.02, fontsize=16, fontweight='bold')

    output_file = os.path.join(output_dir, '06_size_faceted_boxplots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 7: Best Values (Lower is Better) - Log Scale
# ============================================================================
def plot_best_values_boxplot_log(df, output_dir):
    """Create box plot of best values on log scale"""

    plt.figure(figsize=(14, 8))

    # Select algorithms and filter out extreme values
    top_algos = df.groupby('algo_clean')['mean_best'].mean().nsmallest(15).index
    df_top = df[df['algo_clean'].isin(top_algos)].copy()

    # Add small constant to avoid log(0)
    df_top['mean_best_adj'] = df_top['mean_best'] + 1e-10

    # Create box plot with log scale
    ax = sns.boxplot(data=df_top, x='algo_clean', y='mean_best_adj',
                     palette='magma', width=0.7)
    ax.set_yscale('log')

    plt.title('Distribution of Best Values (Lower is Better)\nLog Scale Box Plot',
              fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Best Value (log scale)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '07_best_values_log_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 8: Paired Box Plot (Binary vs Continuous)
# ============================================================================
def plot_paired_domain_boxplot(df, output_dir):
    """Create paired box plots comparing binary and continuous problems"""

    # Separate problem types
    binary_domains = ['onemax', 'trap5', 'leadingones', 'knapsack01']
    continuous_domains = ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'griewank', 'levy']

    df['problem_type'] = 'other'
    df.loc[df['domain'].isin(binary_domains), 'problem_type'] = 'binary'
    df.loc[df['domain'].isin(continuous_domains), 'problem_type'] = 'continuous'

    # Filter out other
    df_typed = df[df['problem_type'] != 'other'].copy()

    # Select top algorithms
    top_algos = df_typed.groupby('algo_clean')['succ_rate'].mean().nlargest(10).index

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Binary problems
    df_binary = df_typed[(df_typed['problem_type'] == 'binary') &
                         (df_typed['algo_clean'].isin(top_algos))]
    sns.boxplot(data=df_binary, x='algo_clean', y='succ_rate',
                ax=axes[0], palette='Blues', width=0.7)
    axes[0].set_title('Binary Problems', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Algorithm')
    axes[0].set_ylabel('Success Rate')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim(0, 1)

    # Continuous problems
    df_cont = df_typed[(df_typed['problem_type'] == 'continuous') &
                       (df_typed['algo_clean'].isin(top_algos))]
    sns.boxplot(data=df_cont, x='algo_clean', y='succ_rate',
                ax=axes[1], palette='Reds', width=0.7)
    axes[1].set_title('Continuous Problems', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Algorithm')
    axes[1].set_ylabel('Success Rate')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1)

    plt.suptitle('Algorithm Performance: Binary vs Continuous Problems\n(Paired Box Plots)',
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()

    output_file = os.path.join(output_dir, '08_paired_domain_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 9: Custom Box Plot with Statistical Annotations
# ============================================================================
def plot_annotated_stats_boxplot(df, output_dir):
    """Create box plot with statistical annotations"""

    plt.figure(figsize=(14, 8))

    # Select top 8 algorithms
    top_8 = df.groupby('algo_clean')['succ_rate'].mean().nlargest(8).index
    df_top = df[df['algo_clean'].isin(top_8)]

    # Create box plot
    bp = sns.boxplot(data=df_top, x='algo_clean', y='succ_rate',
                     palette='pastel', width=0.6)

    # Add statistical annotations
    for i, algo in enumerate(top_8):
        algo_data = df_top[df_top['algo_clean'] == algo]['succ_rate']
        median = algo_data.median()
        q1 = algo_data.quantile(0.25)
        q3 = algo_data.quantile(0.75)
        iqr = q3 - q1

        # Annotate median
        plt.text(i, median + 0.02, f'Med={median:.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Annotate IQR
        plt.text(i, q3 + 0.04, f'IQR={iqr:.2f}',
                 ha='center', va='bottom', fontsize=8, color='gray')

        # Count samples
        n_samples = len(algo_data)
        plt.text(i, -0.05, f'n={n_samples}',
                 ha='center', va='top', fontsize=9, style='italic')

    plt.title('Box Plot with Statistical Annotations\n(Median, IQR, Sample Size)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '09_annotated_stats_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 10: Swarm + Box Plot (Better visualization of distribution)
# ============================================================================
def plot_boxplot_with_swarm(df, output_dir):
    """Combine box plot with swarm plot for detailed distribution"""

    plt.figure(figsize=(14, 8))

    # Select top algorithms
    top_algos = df.groupby('algo_clean')['succ_rate'].mean().nlargest(10).index
    df_top = df[df['algo_clean'].isin(top_algos)]

    # Create box plot
    sns.boxplot(data=df_top, x='algo_clean', y='succ_rate',
                palette='light:#5A9', width=0.7, linewidth=1.5,
                fliersize=0)  # Hide outliers

    # Add swarm plot for individual points
    sns.swarmplot(data=df_top, x='algo_clean', y='succ_rate',
                  color='black', alpha=0.6, size=4)

    plt.title('Box Plot with Swarm Overlay\n(Shows all individual data points)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '10_boxplot_with_swarm.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 11: Letter-value Plot (Extended box plot for large data)
# ============================================================================
def plot_letter_value_plot(df, output_dir):
    """
    Letter-value plot - shows more quantiles than box plot
    Requires 'proplot' package: pip install proplot
    """
    try:
        import proplot as pplt

        fig, ax = pplt.subplots(figsize=(12, 6))

        # Select top algorithms
        top_algos = df.groupby('algo_clean')['succ_rate'].mean().nlargest(10).index
        df_top = df[df['algo_clean'].isin(top_algos)]

        # Pivot data
        pivot_data = df_top.pivot(columns='algo_clean', values='succ_rate')

        # Create letter-value plot
        ax.lvplot(pivot_data, marker='s', markerfacecolor='k',
                  markeredgecolor='k', markersize=4)

        ax.format(title='Letter-Value Plot (Extended Box Plot)',
                  xlabel='Algorithm', ylabel='Success Rate',
                  xticklabels=top_algos, grid=True)

        plt.suptitle('Shows more quantiles than standard box plot',
                     y=0.98, fontsize=12)

        output_file = os.path.join(output_dir, '11_letter_value_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    except ImportError:
        print("Skipping letter-value plot - install 'proplot' package")


# ============================================================================
# BOX PLOT TYPE 12: Bax plot with confidence intervals (bootstrapped)
# ============================================================================
def plot_bootstrapped_boxplot(df, output_dir):
    """Create box plot with bootstrapped confidence intervals"""

    from scipy import stats

    plt.figure(figsize=(14, 8))

    # Select top 8 algorithms
    top_8 = df.groupby('algo_clean')['succ_rate'].mean().nlargest(8).index
    df_top = df[df['algo_clean'].isin(top_8)]

    # Create box plot
    bp = sns.boxplot(data=df_top, x='algo_clean', y='succ_rate',
                     palette='Set2', width=0.6)

    # Add bootstrapped 95% CI for means
    for i, algo in enumerate(top_8):
        algo_data = df_top[df_top['algo_clean'] == algo]['succ_rate'].dropna()

        if len(algo_data) > 1:
            # Bootstrap confidence interval
            n_bootstrap = 1000
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(algo_data, size=len(algo_data), replace=True)
                bootstrap_means.append(sample.mean())

            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)

            # Plot CI
            plt.plot([i - 0.2, i + 0.2], [ci_lower, ci_lower], 'b-', linewidth=1)
            plt.plot([i - 0.2, i + 0.2], [ci_upper, ci_upper], 'b-', linewidth=1)
            plt.plot([i, i], [ci_lower, ci_upper], 'b-', linewidth=1)
            plt.plot(i, algo_data.mean(), 'bD', markersize=6)

    plt.title('Box Plot with Bootstrapped 95% Confidence Intervals\n(Blue diamonds = means, blue lines = CI)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '12_bootstrapped_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 13: Relative Performance Box Plot (normalized to best)
# ============================================================================
def plot_relative_performance_boxplot(df, output_dir):
    """Create box plot showing relative performance compared to best algorithm"""

    plt.figure(figsize=(14, 8))

    # Select top algorithms
    top_algos = df.groupby('algo_clean')['succ_rate'].mean().nlargest(15).index
    df_top = df[df['algo_clean'].isin(top_algos)].copy()

    # Calculate relative performance (compared to max success rate)
    max_success = df_top.groupby('instance_id')['succ_rate'].transform('max')
    df_top['relative_perf'] = df_top['succ_rate'] / max_success
    df_top['relative_perf'] = df_top['relative_perf'].fillna(0)

    # Create box plot
    sns.boxplot(data=df_top, x='algo_clean', y='relative_perf',
                palette='RdYlGn', width=0.7)

    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Best Performance')

    plt.title('Relative Performance Box Plot\n(1.0 = best algorithm for that problem)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Relative Performance (higher is better)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '13_relative_performance_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 14: Time-based Box Plot (if budget is considered as time)
# ============================================================================
def plot_budget_grouped_boxplot(df, output_dir):
    """Create box plot grouped by budget levels"""

    # Create budget categories
    df['budget_category'] = pd.cut(df['budget'],
                                   bins=[0, 500, 2000, 5000, float('inf')],
                                   labels=['Low (≤500)', 'Medium (501-2000)',
                                           'High (2001-5000)', 'Very High (>5000)'])

    # Select top algorithms
    top_algos = df.groupby('algo_clean')['succ_rate'].mean().nlargest(8).index
    df_top = df[df['algo_clean'].isin(top_algos)]

    plt.figure(figsize=(16, 8))

    # Create grouped box plot
    sns.boxplot(data=df_top, x='algo_clean', y='succ_rate',
                hue='budget_category', palette='viridis', width=0.8)

    plt.title('Algorithm Performance Across Different Budget Levels\n(Grouped by Budget)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Budget Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '14_budget_grouped_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# BOX PLOT TYPE 15: Publication-Ready Box Plot (minimalist, high quality)
# ============================================================================
def plot_publication_boxplot(df, output_dir):
    """Create publication-ready minimalist box plot"""

    # Use publication style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (12, 6)
    })

    fig, ax = plt.subplots()

    # Select top 8 algorithms
    top_8 = df.groupby('algo_clean')['succ_rate'].mean().nlargest(8).index
    df_top = df[df['algo_clean'].isin(top_8)]

    # Create minimalist box plot
    box = ax.boxplot([df_top[df_top['algo_clean'] == algo]['succ_rate'].dropna()
                      for algo in top_8],
                     labels=top_8,
                     patch_artist=True,
                     showmeans=True,
                     meanline=True,
                     medianprops={'color': 'black', 'linewidth': 1.5},
                     meanprops={'color': 'red', 'linewidth': 1.5, 'linestyle': '--'},
                     whiskerprops={'color': 'gray', 'linewidth': 1},
                     capprops={'color': 'gray', 'linewidth': 1},
                     boxprops={'facecolor': 'lightgray', 'alpha': 0.7})

    ax.set_title('Publication-Ready Box Plot\nAlgorithm Performance Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, linestyle=':')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='black', lw=1.5, label='Median'),
                       Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='Mean')]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_file = os.path.join(output_dir, '15_publication_boxplot.png')
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main function to generate all box plots"""

    if len(sys.argv) < 2:
        print("Usage: python boxplot_generator.py <data_file.csv>")
        print("\nExample: python boxplot_generator.py optimization_results.csv")
        sys.exit(1)

    data_file = sys.argv[1]

    if not os.path.exists(data_file):
        print(f"Error: File '{data_file}' not found!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("BOX PLOT GENERATOR FOR OPTIMIZATION BENCHMARKS")
    print("=" * 60 + "\n")

    # Load data
    df = load_data(data_file)

    # Create output directory
    output_dir = create_output_dir()

    # Generate all box plot types
    print(f"\nGenerating 15 different box plot visualizations...\n")

    plot_success_rate_boxplot(df, output_dir)
    plot_domain_grouped_boxplot(df, output_dir)
    plot_violin_distribution(df, output_dir)
    plot_horizontal_boxplot_with_means(df, output_dir)
    plot_notched_boxplot(df, output_dir)
    plot_size_faceted_boxplots(df, output_dir)
    plot_best_values_boxplot_log(df, output_dir)
    plot_paired_domain_boxplot(df, output_dir)
    plot_annotated_stats_boxplot(df, output_dir)
    plot_boxplot_with_swarm(df, output_dir)
    plot_letter_value_plot(df, output_dir)
    plot_bootstrapped_boxplot(df, output_dir)
    plot_relative_performance_boxplot(df, output_dir)
    plot_budget_grouped_boxplot(df, output_dir)
    plot_publication_boxplot(df, output_dir)

    print(f"\n✅ Successfully generated 15 box plots in '{output_dir}/'")
    print("\nFiles created:")
    print("  01_success_rate_boxplot.png          - Standard box plot of success rates")
    print("  02_domain_grouped_boxplot.png        - Grouped by problem domain")
    print("  03_violin_distribution.png           - Violin plot with distribution shape")
    print("  04_horizontal_boxplot.png            - Horizontal layout with mean markers")
    print("  05_notched_boxplot.png               - Notched box plot (confidence intervals)")
    print("  06_size_faceted_boxplots.png         - Faceted by problem size")
    print("  07_best_values_log_boxplot.png       - Best values on log scale")
    print("  08_paired_domain_boxplot.png         - Binary vs continuous comparison")
    print("  09_annotated_stats_boxplot.png       - With statistical annotations")
    print("  10_boxplot_with_swarm.png            - Combined with swarm plot")
    print("  11_letter_value_plot.png             - Extended quantile plot")
    print("  12_bootstrapped_boxplot.png          - With bootstrapped confidence intervals")
    print("  13_relative_performance_boxplot.png  - Relative to best algorithm")
    print("  14_budget_grouped_boxplot.png        - Grouped by computational budget")
    print("  15_publication_boxplot.png           - Minimalist publication-ready")


if __name__ == "__main__":
    main()