import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("PORTFOLIO PERFORMANCE VISUALIZATION")
print("="*60)

# Load the backtesting results
if not os.path.exists('portfolio_backtest_timeseries.csv'):
    print("❌ ERROR: Run portfolio_backtest_complete.py first!")
    exit(1)

# Load data
results = pd.read_csv('portfolio_backtest_timeseries.csv', parse_dates=['date'])
results = results.sort_values('date').reset_index(drop=True)

print(f"✓ Loaded {len(results)} months of data")
print(f"   Date range: {results['date'].min().strftime('%Y-%m')} to {results['date'].max().strftime('%Y-%m')}")

# Create figure with subplots
fig = plt.figure(figsize=(20, 16))

# 1. CUMULATIVE RETURNS COMPARISON
ax1 = plt.subplot(3, 2, 1)
strategies = ['long_short_return', 'long_portfolio', 'short_portfolio', 'sp_ret']
strategy_labels = ['Long-Short Strategy', 'Long Portfolio', 'Short Portfolio', 'S&P 500']
colors = ['red', 'green', 'orange', 'blue']

for strategy, label, color in zip(strategies, strategy_labels, colors):
    if strategy in results.columns:
        cumulative = (1 + results[strategy]).cumprod()
        plt.plot(results['date'], cumulative, label=label, linewidth=2, color=color)

plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to better see relative performance

# 2. ROLLING SHARPE RATIO (12-month window)
ax2 = plt.subplot(3, 2, 2)
window = 12
for strategy, label, color in zip(strategies[:3], strategy_labels[:3], colors[:3]):
    if strategy in results.columns:
        rolling_mean = results[strategy].rolling(window).mean()
        rolling_std = results[strategy].rolling(window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(12)  # Annualized
        plt.plot(results['date'], rolling_sharpe, label=f'{label}', linewidth=2, color=color)

plt.title(f'Rolling {window}-Month Annualized Sharpe Ratio', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 3. DRAWDOWN ANALYSIS
ax3 = plt.subplot(3, 2, 3)
strategy = 'long_short_return'
if strategy in results.columns:
    cumulative = (1 + results[strategy]).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    plt.fill_between(results['date'], drawdown, 0, alpha=0.3, color='red', label='Drawdown')
    plt.plot(results['date'], drawdown, color='red', linewidth=1)
    
    # Highlight maximum drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_date = results.loc[max_dd_idx, 'date']
    max_dd_value = drawdown.iloc[max_dd_idx]
    plt.scatter([max_dd_date], [max_dd_value], color='darkred', s=100, zorder=5)
    plt.annotate(f'Max DD: {max_dd_value:.2%}\n{max_dd_date.strftime("%Y-%m")}', 
                xy=(max_dd_date, max_dd_value), xytext=(10, 10),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.title('Long-Short Strategy Drawdown', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# 4. MONTHLY RETURNS DISTRIBUTION
ax4 = plt.subplot(3, 2, 4)
strategy = 'long_short_return'
if strategy in results.columns:
    returns = results[strategy]
    
    # Histogram
    plt.hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # Add normal distribution overlay
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    plt.plot(x, normal_dist, 'r-', linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
    
    # Add vertical lines for mean and median
    plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.3f}')
    plt.axvline(returns.median(), color='green', linestyle='--', label=f'Median: {returns.median():.3f}')

plt.title('Long-Short Strategy Monthly Returns Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Monthly Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. ROLLING CORRELATION WITH MARKET
ax5 = plt.subplot(3, 2, 5)
window = 24
if 'long_short_return' in results.columns and 'sp_ret' in results.columns:
    rolling_corr = results['long_short_return'].rolling(window).corr(results['sp_ret'])
    plt.plot(results['date'], rolling_corr, linewidth=2, color='purple')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='High Correlation')
    plt.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5)

plt.title(f'Rolling {window}-Month Correlation with S&P 500', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-1, 1)

# 6. PORTFOLIO PERFORMANCE METRICS COMPARISON
ax6 = plt.subplot(3, 2, 6)

# Load summary metrics if available
if os.path.exists('portfolio_backtest_results.json'):
    with open('portfolio_backtest_results.json', 'r') as f:
        summary = json.load(f)
    
    # Use the correct key structure
    strategies = ['strategy', 'long_only', 'market']
    strategy_names = ['Long-Short', 'Long Only', 'S&P 500']
    
    metrics = ['annualized_return', 'annualized_volatility', 'annualized_sharpe', 'max_drawdown']
    metric_labels = ['Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Max Drawdown']
    
    # Create comparison bar chart
    x = np.arange(len(metric_labels))
    width = 0.25
    
    for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        if strategy in summary['performance_metrics']:
            values = [summary['performance_metrics'][strategy][metric] for metric in metrics]
            # Convert drawdown to positive for better visualization
            values[3] = abs(values[3])
            plt.bar(x + i*width, values, width, label=name, alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x + width, metric_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('portfolio_performance_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Main performance chart saved as 'portfolio_performance_analysis.png'")

# CREATE ADDITIONAL DETAILED CHARTS
print("\nCreating additional detailed charts...")

# CHART 2: TURNOVER ANALYSIS
if os.path.exists('portfolio_turnover_analysis.csv'):
    turnover_df = pd.read_csv('portfolio_turnover_analysis.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Monthly turnover over time
    turnover_df['date'] = pd.to_datetime(turnover_df[['year', 'month']].assign(day=1))
    
    # Plot long and short turnover separately
    ax1.plot(turnover_df['date'], turnover_df['long_turnover'], 
            label='Long Portfolio Turnover', linewidth=2, marker='o', markersize=3, color='green')
    ax1.plot(turnover_df['date'], turnover_df['short_turnover'], 
            label='Short Portfolio Turnover', linewidth=2, marker='s', markersize=3, color='red')
    ax1.plot(turnover_df['date'], turnover_df['overall_turnover'], 
            label='Overall Turnover', linewidth=2, marker='^', markersize=3, color='blue')
    
    ax1.set_title('Monthly Portfolio Turnover Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Turnover Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Average turnover comparison
    avg_long_turnover = turnover_df['long_turnover'].mean()
    avg_short_turnover = turnover_df['short_turnover'].mean()
    avg_overall_turnover = turnover_df['overall_turnover'].mean()
    
    categories = ['Long Portfolio', 'Short Portfolio', 'Overall Strategy']
    values = [avg_long_turnover, avg_short_turnover, avg_overall_turnover]
    colors = ['green', 'red', 'blue']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7)
    ax2.set_title('Average Monthly Turnover by Portfolio Type', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Turnover Rate')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('portfolio_turnover_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Turnover analysis chart saved as 'portfolio_turnover_analysis.png'")

# CHART 3: RISK-RETURN SCATTER PLOT
fig, ax = plt.subplots(figsize=(10, 8))

if os.path.exists('portfolio_backtest_results.json'):
    with open('portfolio_backtest_results.json', 'r') as f:
        summary = json.load(f)
    
    strategies = ['strategy', 'long_only', 'short_only', 'market']
    strategy_names = ['Long-Short Strategy', 'Long Portfolio', 'Short Portfolio', 'S&P 500']
    colors = ['red', 'green', 'orange', 'blue']
    
    for strategy, name, color in zip(strategies, strategy_names, colors):
        if strategy in summary['performance_metrics']:
            x = summary['performance_metrics'][strategy]['annualized_volatility']
            y = summary['performance_metrics'][strategy]['annualized_return']
            plt.scatter(x, y, s=200, alpha=0.7, color=color, label=name, edgecolors='black')
            
            # Add labels
            plt.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold')

# Add efficient frontier reference lines
x_range = np.linspace(0, plt.xlim()[1], 100)
for sharpe in [0.5, 1.0, 1.5]:
    y_range = sharpe * x_range
    plt.plot(x_range, y_range, '--', alpha=0.3, color='gray', 
            label=f'Sharpe = {sharpe}' if sharpe == 1.0 else '')

plt.xlabel('Annualized Volatility', fontsize=12)
plt.ylabel('Annualized Return', fontsize=12)
plt.title('Risk-Return Profile Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.tight_layout()
plt.savefig('risk_return_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Risk-return analysis chart saved as 'risk_return_analysis.png'")

# SUMMARY STATISTICS TABLE
print(f"\n📊 VISUALIZATION SUMMARY:")
print("-" * 40)
print("Generated charts:")
print("1. portfolio_performance_analysis.png - Main 6-panel performance dashboard")
print("2. portfolio_turnover_analysis.png - Portfolio turnover analysis")
print("3. risk_return_analysis.png - Risk-return scatter plot")

if os.path.exists('portfolio_backtest_results.json'):
    with open('portfolio_backtest_results.json', 'r') as f:
        summary = json.load(f)
    
    ls_metrics = summary['performance_metrics']['strategy']
    print(f"\n🎯 KEY PERFORMANCE HIGHLIGHTS:")
    print(f"   • Strategy Return: {ls_metrics['annualized_return']*100:.1f}% per year")
    print(f"   • Strategy Volatility: {ls_metrics['annualized_volatility']*100:.1f}% per year")
    print(f"   • Sharpe Ratio: {ls_metrics['annualized_sharpe']:.2f}")
    print(f"   • Alpha vs Market: {ls_metrics['alpha_annualized']*100:.1f}% per year")
    print(f"   • Maximum Drawdown: {abs(ls_metrics['max_drawdown'])*100:.1f}%")
    print(f"   • Win Rate: {ls_metrics['win_rate']*100:.1f}%")

plt.show()
print(f"\n✅ All visualizations complete!")
print("="*60) 