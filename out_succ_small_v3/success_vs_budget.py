import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
import warnings

warnings.filterwarnings('ignore')

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('instance_algo_budget_summary.csv')

# Filter out rows with NaN in successes or trials
df_clean = df.dropna(subset=['successes', 'trials'])

# Calculate empirical success rate and posterior mean
# Using Beta(1,1) prior (uniform), posterior mean = (successes + 1) / (trials + 2)
df_clean['posterior_mean'] = (df_clean['successes'] + 1) / (df_clean['trials'] + 2)

# Separate binary and continuous domains
df_binary = df_clean[df_clean['domain'] == 'bin'].copy()
df_continuous = df_clean[df_clean['domain'].str.contains('cont', na=False)].copy()


# Function to process data for a given domain and tier
def process_tier_data(df, tier):
    tier_data = df[df['target_name'] == tier].copy()

    if len(tier_data) == 0:
        return None

    # Group by budget and calculate statistics
    budget_stats = tier_data.groupby('budget').agg({
        'posterior_mean': ['mean', 'std', 'count']
    }).reset_index()
    budget_stats.columns = ['budget', 'mean_posterior', 'std_posterior', 'count']

    # Sort by budget
    budget_stats = budget_stats.sort_values('budget')

    # Calculate confidence intervals (95% CI using normal approximation)
    budget_stats['ci_lower'] = budget_stats['mean_posterior'] - 1.96 * budget_stats['std_posterior'] / np.sqrt(
        budget_stats['count'])
    budget_stats['ci_upper'] = budget_stats['mean_posterior'] + 1.96 * budget_stats['std_posterior'] / np.sqrt(
        budget_stats['count'])

    # Clip CI to [0,1] range
    budget_stats['ci_lower'] = budget_stats['ci_lower'].clip(0, 1)
    budget_stats['ci_upper'] = budget_stats['ci_upper'].clip(0, 1)

    return budget_stats, tier_data


# Create combined figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Mean Posterior Success Probability vs Budget\nBinary and Continuous Domains by Target Tier',
             fontsize=16, fontweight='bold')

# Define colors for different domains
domain_colors = {'Binary': 'blue', 'Continuous': 'green'}

# Process and plot each domain and tier
tiers = ['easy', 'med', 'hard']

for row_idx, (domain_name, domain_df) in enumerate([('Binary', df_binary), ('Continuous', df_continuous)]):
    for col_idx, tier in enumerate(tiers):
        ax = axes[row_idx, col_idx]

        # Process data for this domain and tier
        result = process_tier_data(domain_df, tier)

        if result is None:
            ax.text(0.5, 0.5, f'No data for {tier} tier',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{domain_name} - {tier.capitalize()} Tier')
            continue

        budget_stats, tier_data = result

        # Plot main line with confidence interval
        ax.plot(budget_stats['budget'], budget_stats['mean_posterior'],
                'o-', linewidth=2, markersize=8,
                color=domain_colors[domain_name],
                label=f'{domain_name} (n={len(tier_data)})')
        ax.fill_between(budget_stats['budget'],
                        budget_stats['ci_lower'],
                        budget_stats['ci_upper'],
                        alpha=0.2, color=domain_colors[domain_name])

        # Add scatter points for individual algorithms (with low opacity)
        for budget in budget_stats['budget'].unique():
            budget_points = tier_data[tier_data['budget'] == budget]
            ax.scatter([budget] * len(budget_points),
                       budget_points['posterior_mean'],
                       alpha=0.2, s=20, c=domain_colors[domain_name], marker='.')

        # Customize plot
        ax.set_xlabel('Budget', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel('Mean Posterior Success Probability', fontsize=11)

        ax.set_title(f'{domain_name} - {tier.capitalize()} Tier', fontsize=13, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Add text with budget values
        if len(budget_stats) > 0:
            budgets_text = f"Budgets: {', '.join([str(int(b)) for b in budget_stats['budget'].values])}"
            ax.text(0.02, 0.98, budgets_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure_3_1_combined.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS BY DOMAIN AND TIER")
print("=" * 80)

for domain_name, domain_df in [('Binary', df_binary), ('Continuous', df_continuous)]:
    print(f"\n{domain_name} Domain:")
    print("-" * 40)

    for tier in tiers:
        result = process_tier_data(domain_df, tier)
        if result is not None:
            budget_stats, tier_data = result
            print(f"\n  {tier.capitalize()} Tier (n={len(tier_data)}):")
            for _, row in budget_stats.iterrows():
                print(f"    Budget {int(row['budget'])}: mean={row['mean_posterior']:.3f}, "
                      f"95% CI=[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], n={int(row['count'])}")

# Create a comparison boxplot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Distribution Comparison: Binary vs Continuous Domains by Tier',
             fontsize=14, fontweight='bold')

for idx, tier in enumerate(tiers):
    ax = axes[idx]

    # Prepare data for boxplot
    binary_data = []
    continuous_data = []
    budgets = []

    binary_result = process_tier_data(df_binary, tier)
    continuous_result = process_tier_data(df_continuous, tier)

    if binary_result is not None:
        binary_stats, binary_raw = binary_result
        budgets = sorted(binary_stats['budget'].unique())

        # Create boxplot data
        bp_data = []
        positions = []
        labels = []

        for i, budget in enumerate(budgets):
            # Binary data at this budget
            binary_budget_data = binary_raw[binary_raw['budget'] == budget]['posterior_mean'].values
            bp_data.append(binary_budget_data)
            positions.append(i * 3)  # Space out binary and continuous
            labels.append(f'B-{int(budget)}')

            # Continuous data at this budget (if exists)
            if continuous_result is not None:
                continuous_budget_data = continuous_result[1][continuous_result[1]['budget'] == budget][
                    'posterior_mean'].values
                if len(continuous_budget_data) > 0:
                    bp_data.append(continuous_budget_data)
                    positions.append(i * 3 + 1)
                    labels.append(f'C-{int(budget)}')

        # Create boxplot
        if len(bp_data) > 0:
            bp = ax.boxplot(bp_data, positions=positions, widths=0.6,
                            patch_artist=True, showmeans=True,
                            meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6})

            # Color boxes
            for j, box in enumerate(bp['boxes']):
                if j % 2 == 0:  # Binary
                    box.set_facecolor('lightblue')
                else:  # Continuous
                    box.set_facecolor('lightgreen')
                box.set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Posterior Success Probability')
            ax.set_title(f'{tier.capitalize()} Tier')

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='Binary'),
                               Patch(facecolor='lightgreen', alpha=0.7, label='Continuous')]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('figure_3_1_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnalysis complete! Generated files:")
print("  - figure_3_1_combined.png (Main figure with both domains)")
print("  - figure_3_1_comparison.png (Comparison boxplots)")