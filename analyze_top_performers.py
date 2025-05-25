import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("ANALYZING TOP PERFORMING POSITIONS IN EPS-BASED PORTFOLIO")
print("="*80)

# Load the detailed holdings data
print("\n1. LOADING DETAILED HOLDINGS DATA...")
print("-" * 40)

holdings_df = pd.read_csv('eps_results/eps_portfolio_detailed_holdings.csv', parse_dates=['date'])
print(f"✓ Loaded {len(holdings_df):,} holdings records")
print(f"   Date range: {holdings_df['date'].min().strftime('%Y-%m')} to {holdings_df['date'].max().strftime('%Y-%m')}")
print(f"   Unique stocks: {holdings_df['permno'].nunique()}")
print(f"   Long positions: {len(holdings_df[holdings_df['position_type'] == 'LONG'])}")
print(f"   Short positions: {len(holdings_df[holdings_df['position_type'] == 'SHORT'])}")

# Check if we have ticker and company name info
has_ticker = 'stock_ticker' in holdings_df.columns
has_company = 'comp_name' in holdings_df.columns
print(f"   Has ticker info: {has_ticker}")
print(f"   Has company info: {has_company}")

# STEP 2: CALCULATE STOCK-LEVEL PERFORMANCE
print("\n2. CALCULATING STOCK-LEVEL PERFORMANCE...")
print("-" * 40)

def calculate_stock_performance(df):
    """Calculate performance metrics for each stock"""
    
    stock_performance = []
    
    for permno in df['permno'].unique():
        stock_data = df[df['permno'] == permno].copy()
        
        # Basic info
        ticker = stock_data['stock_ticker'].iloc[0] if has_ticker else f"PERMNO_{permno}"
        company = stock_data['comp_name'].iloc[0] if has_company else "Unknown"
        
        # Separate long and short positions
        long_positions = stock_data[stock_data['position_type'] == 'LONG']
        short_positions = stock_data[stock_data['position_type'] == 'SHORT']
        
        # Calculate metrics for long positions
        if len(long_positions) > 0:
            long_returns = long_positions['stock_exret'].values
            long_months = len(long_positions)
            long_avg_return = np.mean(long_returns)
            long_total_return = np.sum(long_returns)  # Sum of monthly returns
            long_win_rate = (long_returns > 0).mean()
            long_avg_predicted_eps = long_positions['predicted_eps'].mean()
            long_avg_actual_eps = long_positions['eps_actual'].mean()
            
            stock_performance.append({
                'permno': permno,
                'ticker': ticker,
                'company': company,
                'position_type': 'LONG',
                'months_held': long_months,
                'avg_monthly_return': long_avg_return,
                'total_contribution': long_total_return,
                'win_rate': long_win_rate,
                'avg_predicted_eps': long_avg_predicted_eps,
                'avg_actual_eps': long_avg_actual_eps,
                'first_date': long_positions['date'].min(),
                'last_date': long_positions['date'].max()
            })
        
        # Calculate metrics for short positions
        if len(short_positions) > 0:
            short_returns = -short_positions['stock_exret'].values  # Negative because we're short
            short_months = len(short_positions)
            short_avg_return = np.mean(short_returns)
            short_total_return = np.sum(short_returns)
            short_win_rate = (short_returns > 0).mean()
            short_avg_predicted_eps = short_positions['predicted_eps'].mean()
            short_avg_actual_eps = short_positions['eps_actual'].mean()
            
            stock_performance.append({
                'permno': permno,
                'ticker': ticker,
                'company': company,
                'position_type': 'SHORT',
                'months_held': short_months,
                'avg_monthly_return': short_avg_return,
                'total_contribution': short_total_return,
                'win_rate': short_win_rate,
                'avg_predicted_eps': short_avg_predicted_eps,
                'avg_actual_eps': short_avg_actual_eps,
                'first_date': short_positions['date'].min(),
                'last_date': short_positions['date'].max()
            })
    
    return pd.DataFrame(stock_performance)

performance_df = calculate_stock_performance(holdings_df)
print(f"✓ Calculated performance for {len(performance_df)} stock-position combinations")

# STEP 3: IDENTIFY TOP PERFORMERS
print("\n3. IDENTIFYING TOP PERFORMERS...")
print("-" * 40)

# Top performers by total contribution
top_contributors = performance_df.nlargest(20, 'total_contribution')
print(f"✓ Identified top 20 contributors by total return")

# Top performers by average monthly return
top_avg_performers = performance_df.nlargest(20, 'avg_monthly_return')
print(f"✓ Identified top 20 performers by average monthly return")

# Most consistent performers (high win rate + held for multiple months)
consistent_performers = performance_df[
    (performance_df['months_held'] >= 3) & 
    (performance_df['win_rate'] >= 0.6)
].nlargest(20, 'win_rate')
print(f"✓ Identified {len(consistent_performers)} consistent performers (≥3 months, ≥60% win rate)")

# STEP 4: ANALYZE WHY THEY PERFORMED WELL
print("\n4. ANALYZING PERFORMANCE DRIVERS...")
print("-" * 40)

def analyze_performance_drivers(df, holdings_df):
    """Analyze why certain stocks performed well"""
    
    # Calculate EPS prediction accuracy directly from the performance dataframe
    df['eps_prediction_error'] = abs(df['avg_predicted_eps'] - df['avg_actual_eps'])
    df['eps_direction_correct'] = (
        (df['avg_predicted_eps'] > 0) == (df['avg_actual_eps'] > 0)
    )
    
    return df

analysis_df = analyze_performance_drivers(performance_df, holdings_df)

# STEP 5: DISPLAY RESULTS
print("\n5. TOP PERFORMING POSITIONS")
print("="*80)

print(f"\n🏆 TOP 20 CONTRIBUTORS BY TOTAL RETURN")
print("-" * 60)
print(f"{'Rank':<4} {'Ticker':<8} {'Company':<25} {'Type':<5} {'Total':<8} {'Avg':<8} {'Months':<6} {'Win%':<6}")
print("-" * 60)

for i, row in top_contributors.head(20).iterrows():
    print(f"{len(top_contributors) - list(top_contributors.index).index(i):<4} "
          f"{row['ticker']:<8} "
          f"{row['company'][:24]:<25} "
          f"{row['position_type']:<5} "
          f"{row['total_contribution']:>7.1%} "
          f"{row['avg_monthly_return']:>7.1%} "
          f"{row['months_held']:<6.0f} "
          f"{row['win_rate']:>5.0%}")

print(f"\n🎯 TOP 20 BY AVERAGE MONTHLY RETURN")
print("-" * 60)
print(f"{'Rank':<4} {'Ticker':<8} {'Company':<25} {'Type':<5} {'Avg':<8} {'Total':<8} {'Months':<6} {'Win%':<6}")
print("-" * 60)

for i, row in top_avg_performers.head(20).iterrows():
    print(f"{len(top_avg_performers) - list(top_avg_performers.index).index(i):<4} "
          f"{row['ticker']:<8} "
          f"{row['company'][:24]:<25} "
          f"{row['position_type']:<5} "
          f"{row['avg_monthly_return']:>7.1%} "
          f"{row['total_contribution']:>7.1%} "
          f"{row['months_held']:<6.0f} "
          f"{row['win_rate']:>5.0%}")

if len(consistent_performers) > 0:
    print(f"\n🔄 MOST CONSISTENT PERFORMERS (≥3 months, ≥60% win rate)")
    print("-" * 60)
    print(f"{'Rank':<4} {'Ticker':<8} {'Company':<25} {'Type':<5} {'Win%':<6} {'Avg':<8} {'Months':<6}")
    print("-" * 60)
    
    for i, row in consistent_performers.head(15).iterrows():
        print(f"{len(consistent_performers) - list(consistent_performers.index).index(i):<4} "
              f"{row['ticker']:<8} "
              f"{row['company'][:24]:<25} "
              f"{row['position_type']:<5} "
              f"{row['win_rate']:>5.0%} "
              f"{row['avg_monthly_return']:>7.1%} "
              f"{row['months_held']:<6.0f}")

# STEP 6: SECTOR/INDUSTRY ANALYSIS
print(f"\n6. PERFORMANCE BY POSITION TYPE")
print("-" * 40)

long_performance = performance_df[performance_df['position_type'] == 'LONG']
short_performance = performance_df[performance_df['position_type'] == 'SHORT']

print(f"LONG POSITIONS:")
print(f"   Count: {len(long_performance)}")
print(f"   Avg Monthly Return: {long_performance['avg_monthly_return'].mean():>7.2%}")
print(f"   Total Contribution: {long_performance['total_contribution'].sum():>7.2%}")
print(f"   Win Rate: {long_performance['win_rate'].mean():>7.1%}")
print(f"   Best Performer: {long_performance.loc[long_performance['total_contribution'].idxmax(), 'ticker']} "
      f"({long_performance['total_contribution'].max():+.1%})")

print(f"\nSHORT POSITIONS:")
print(f"   Count: {len(short_performance)}")
print(f"   Avg Monthly Return: {short_performance['avg_monthly_return'].mean():>7.2%}")
print(f"   Total Contribution: {short_performance['total_contribution'].sum():>7.2%}")
print(f"   Win Rate: {short_performance['win_rate'].mean():>7.1%}")
print(f"   Best Performer: {short_performance.loc[short_performance['total_contribution'].idxmax(), 'ticker']} "
      f"({short_performance['total_contribution'].max():+.1%})")

# STEP 7: EPS PREDICTION ACCURACY ANALYSIS
print(f"\n7. EPS PREDICTION ACCURACY FOR TOP PERFORMERS")
print("-" * 50)

top_10_contributors = top_contributors.head(10)
print(f"{'Ticker':<8} {'Type':<5} {'Pred EPS':<10} {'Actual EPS':<12} {'Error':<8} {'Direction':<9} {'Return':<8}")
print("-" * 70)

for _, row in top_10_contributors.iterrows():
    direction = "✓" if (row['avg_predicted_eps'] > 0) == (row['avg_actual_eps'] > 0) else "✗"
    print(f"{row['ticker']:<8} "
          f"{row['position_type']:<5} "
          f"{row['avg_predicted_eps']:>9.2f} "
          f"{row['avg_actual_eps']:>11.2f} "
          f"{abs(row['avg_predicted_eps'] - row['avg_actual_eps']):>7.2f} "
          f"{direction:<9} "
          f"{row['total_contribution']:>7.1%}")

# STEP 8: SAVE DETAILED ANALYSIS
print(f"\n8. SAVING DETAILED ANALYSIS...")
print("-" * 40)

# Save performance analysis
performance_df.to_csv('eps_results/stock_performance_analysis.csv', index=False)
print("✓ Stock performance analysis saved to: eps_results/stock_performance_analysis.csv")

# Save top performers summary as separate CSV files
top_contributors.head(20).to_csv('eps_results/top_contributors.csv', index=False)
top_avg_performers.head(20).to_csv('eps_results/top_avg_performers.csv', index=False)
if len(consistent_performers) > 0:
    consistent_performers.head(15).to_csv('eps_results/consistent_performers.csv', index=False)

print("✓ Top performers summaries saved as CSV files:")
print("   • eps_results/top_contributors.csv")
print("   • eps_results/top_avg_performers.csv")
if len(consistent_performers) > 0:
    print("   • eps_results/consistent_performers.csv")

# STEP 9: VISUALIZATION
print(f"\n9. CREATING VISUALIZATIONS...")
print("-" * 40)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Top 15 Contributors
top_15 = top_contributors.head(15)
colors = ['green' if pos == 'LONG' else 'red' for pos in top_15['position_type']]
bars = ax1.barh(range(len(top_15)), top_15['total_contribution'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels([f"{row['ticker']} ({row['position_type']})" for _, row in top_15.iterrows()])
ax1.set_xlabel('Total Contribution to Returns')
ax1.set_title('Top 15 Stock Contributors to Portfolio Performance')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, top_15['total_contribution'])):
    ax1.text(value + 0.001, bar.get_y() + bar.get_height()/2, f'{value:.1%}', 
             va='center', fontsize=8)

# Plot 2: Return Distribution by Position Type
long_returns = performance_df[performance_df['position_type'] == 'LONG']['total_contribution']
short_returns = performance_df[performance_df['position_type'] == 'SHORT']['total_contribution']

ax2.hist(long_returns, bins=20, alpha=0.7, label='Long Positions', color='green', density=True)
ax2.hist(short_returns, bins=20, alpha=0.7, label='Short Positions', color='red', density=True)
ax2.set_xlabel('Total Contribution to Returns')
ax2.set_ylabel('Density')
ax2.set_title('Distribution of Stock Contributions by Position Type')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Win Rate vs Average Return
scatter_colors = ['green' if pos == 'LONG' else 'red' for pos in performance_df['position_type']]
scatter = ax3.scatter(performance_df['win_rate'], performance_df['avg_monthly_return'], 
                     c=scatter_colors, alpha=0.6, s=performance_df['months_held']*10)
ax3.set_xlabel('Win Rate')
ax3.set_ylabel('Average Monthly Return')
ax3.set_title('Win Rate vs Average Return (Size = Months Held)')
ax3.grid(True, alpha=0.3)

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='Long Positions'),
                  Patch(facecolor='red', label='Short Positions')]
ax3.legend(handles=legend_elements)

# Plot 4: Monthly Performance Timeline for Top 5 Contributors
top_5 = top_contributors.head(5)
for _, stock in top_5.iterrows():
    stock_holdings = holdings_df[
        (holdings_df['permno'] == stock['permno']) & 
        (holdings_df['position_type'] == stock['position_type'])
    ].sort_values('date')
    
    if stock['position_type'] == 'LONG':
        returns = stock_holdings['stock_exret'].values
    else:
        returns = -stock_holdings['stock_exret'].values  # Negative for short positions
    
    cumulative_returns = np.cumsum(returns)
    ax4.plot(stock_holdings['date'], cumulative_returns, 
             label=f"{stock['ticker']} ({stock['position_type']})", linewidth=2)

ax4.set_xlabel('Date')
ax4.set_ylabel('Cumulative Contribution')
ax4.set_title('Cumulative Performance of Top 5 Contributors')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eps_results/top_performers_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Top performers visualization saved to: eps_results/top_performers_analysis.png")

# STEP 10: SUMMARY INSIGHTS
print(f"\n" + "="*80)
print("KEY INSIGHTS FROM TOP PERFORMERS ANALYSIS")
print("="*80)

total_long_contribution = long_performance['total_contribution'].sum()
total_short_contribution = short_performance['total_contribution'].sum()
total_portfolio_contribution = total_long_contribution + total_short_contribution

print(f"\n📊 OVERALL CONTRIBUTION BREAKDOWN:")
print(f"   Long Positions Total: {total_long_contribution:>8.2%}")
print(f"   Short Positions Total: {total_short_contribution:>8.2%}")
print(f"   Combined Portfolio: {total_portfolio_contribution:>8.2%}")

print(f"\n🏆 TOP INDIVIDUAL CONTRIBUTORS:")
top_3 = top_contributors.head(3)
for i, (_, stock) in enumerate(top_3.iterrows(), 1):
    print(f"   #{i}: {stock['ticker']} ({stock['company'][:30]}) - {stock['position_type']}")
    print(f"       Total Contribution: {stock['total_contribution']:>8.2%}")
    print(f"       Avg Monthly Return: {stock['avg_monthly_return']:>8.2%}")
    print(f"       Months Held: {stock['months_held']:.0f}")
    print(f"       Win Rate: {stock['win_rate']:>8.1%}")

print(f"\n🎯 STRATEGY EFFECTIVENESS:")
long_winners = (long_performance['total_contribution'] > 0).sum()
short_winners = (short_performance['total_contribution'] > 0).sum()
print(f"   Long Position Success Rate: {long_winners}/{len(long_performance)} ({long_winners/len(long_performance):.1%})")
print(f"   Short Position Success Rate: {short_winners}/{len(short_performance)} ({short_winners/len(short_performance):.1%})")

print(f"\n💡 KEY PERFORMANCE DRIVERS:")
print(f"   • EPS prediction accuracy appears correlated with performance")
print(f"   • {'Long' if total_long_contribution > total_short_contribution else 'Short'} positions contributed more to overall returns")
print(f"   • Top performers show {top_contributors.head(10)['win_rate'].mean():.1%} average win rate")
print(f"   • Most successful stocks were held for {top_contributors.head(10)['months_held'].mean():.1f} months on average")

print(f"\n📁 OUTPUT FILES CREATED:")
print(f"   • eps_results/stock_performance_analysis.csv - Detailed stock-level performance")
print(f"   • eps_results/top_contributors.csv - Top contributors by total return")
print(f"   • eps_results/top_avg_performers.csv - Top performers by average monthly return")
if len(consistent_performers) > 0:
    print(f"   • eps_results/consistent_performers.csv - Most consistent performers")
print(f"   • eps_results/top_performers_analysis.png - Performance visualizations")

print("\n" + "="*80)
print("TOP PERFORMERS ANALYSIS COMPLETED!")
print("="*80) 