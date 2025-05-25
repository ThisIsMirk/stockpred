import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("EPS-BASED PORTFOLIO BACKTESTING - MULTI-ALLOCATION ANALYSIS")
print("="*80)

# CONFIGURATION PARAMETERS
REBALANCE_FREQUENCY = 'monthly'  # Options: 'monthly', 'quarterly', 'semi-annual'
TOTAL_HOLDINGS = 100  # Keep total holdings constant

# Define different allocation strategies to test
ALLOCATION_STRATEGIES = [
    {'long': 100, 'short': 0, 'name': 'Long-Only'},
    {'long': 90, 'short': 10, 'name': '90/10 Long-Short'},
    {'long': 80, 'short': 20, 'name': '80/20 Long-Short'},
    {'long': 70, 'short': 30, 'name': '70/30 Long-Short'},
    {'long': 60, 'short': 40, 'name': '60/40 Long-Short'},
    {'long': 50, 'short': 50, 'name': '50/50 Long-Short'},
    {'long': 40, 'short': 60, 'name': '40/60 Long-Short'},
    {'long': 30, 'short': 70, 'name': '30/70 Long-Short'},
    {'long': 20, 'short': 80, 'name': '20/80 Long-Short'},
    {'long': 10, 'short': 90, 'name': '10/90 Long-Short'},
    {'long': 0, 'short': 100, 'name': 'Short-Only'}
]

print(f"STRATEGY CONFIGURATION:")
print(f"   Strategy: Testing multiple Long/Short allocations")
print(f"   Rebalancing Frequency: {REBALANCE_FREQUENCY}")
print(f"   Total Holdings: {TOTAL_HOLDINGS}")
print(f"   Allocation Strategies: {len(ALLOCATION_STRATEGIES)} different combinations")
print(f"   Range: From Long-Only to Short-Only")

# STEP 1: LOAD BEST EPS MODEL PREDICTIONS
print("\n1. LOADING EPS PREDICTIONS...")
print("-" * 40)

if os.path.exists('eps_predictions/best_eps_model_info.json'):
    with open('eps_predictions/best_eps_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    
    print(f"✓ Best EPS model info found!")
    print(f"   Model: {best_model_info['model_type'].title()}")
    print(f"   Features: {best_model_info['n_features']}")
    print(f"   Overall R²: {best_model_info['overall_r2']:.6f}")
    
    pred_path = best_model_info['predictions_file']
    model = best_model_info['model_type']
    
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        print(f"✓ EPS predictions loaded from {pred_path}")
        
        # Check if ticker/company info is available
        has_ticker = 'stock_ticker' in pred.columns
        has_company = 'comp_name' in pred.columns
        if has_ticker and has_company:
            print("   ✅ Includes ticker symbols and company names")
        elif has_ticker or has_company:
            print("   ⚠️  Partial ticker/company info available")
        else:
            print("   ⚠️  No ticker/company info available")
    else:
        print(f"✗ ERROR: EPS predictions file {pred_path} not found!")
        exit(1)
        
else:
    print("⚠ WARNING: best_eps_model_info.json not found! Using fallback...")
    pred_path = "eps_predictions/eps_predictions_ridge.csv"  
    model = "ridge"
    
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        print(f"✓ Fallback EPS predictions loaded from {pred_path}")
        
        # Check if ticker/company info is available
        has_ticker = 'stock_ticker' in pred.columns
        has_company = 'comp_name' in pred.columns
        if has_ticker and has_company:
            print("   ✅ Includes ticker symbols and company names")
        elif has_ticker or has_company:
            print("   ⚠️  Partial ticker/company info available")
        else:
            print("   ⚠️  No ticker/company info available")
    else:
        print(f"✗ ERROR: No EPS predictions file found!")
        exit(1)

print(f"Data shape: {pred.shape}")
print(f"Date range: {pred['date'].min().strftime('%Y-%m')} to {pred['date'].max().strftime('%Y-%m')}")

# Check if we have stock_exret column for returns
if 'stock_exret' not in pred.columns:
    print("❌ ERROR: 'stock_exret' column not found in EPS predictions!")
    print("   This should be included automatically in the EPS predictions.")
    print("   Please re-run the EPS training script with the updated version.")
    exit(1)
else:
    print("✅ Stock returns already included in EPS predictions!")
    
    # Check for missing stock returns
    missing_returns = pred['stock_exret'].isna().sum()
    if missing_returns > 0:
        print(f"⚠️  {missing_returns} predictions have missing stock returns")
        pred = pred.dropna(subset=['stock_exret'])
        print(f"   Removed missing returns. Final shape: {pred.shape}")

print(f"✓ Ready for backtesting with {len(pred):,} observations")

# STEP 2: LOAD MARKET DATA
print("\n2. LOADING MARKET DATA...")
print("-" * 40)

if os.path.exists('mkt_ind.csv'):
    mkt_data = pd.read_csv('mkt_ind.csv')
    mkt_data['date'] = pd.to_datetime(mkt_data[['year', 'month']].assign(day=1))
    print(f"✓ Market data loaded: {len(mkt_data)} months")
    print(f"   Columns: {list(mkt_data.columns)}")
    print(f"   Date range: {mkt_data['date'].min().strftime('%Y-%m')} to {mkt_data['date'].max().strftime('%Y-%m')}")
else:
    print("✗ ERROR: mkt_ind.csv not found!")
    exit(1)

# STEP 3: DETERMINE REBALANCING PERIODS
print("\n3. SETTING UP REBALANCING SCHEDULE...")
print("-" * 40)

def get_rebalancing_periods(data, frequency):
    """Determine which months to rebalance based on frequency"""
    months = data.groupby(['year', 'month']).first().reset_index()
    months['date'] = pd.to_datetime(months[['year', 'month']].assign(day=1))
    months = months.sort_values('date').reset_index(drop=True)
    
    rebalance_months = []
    
    if frequency == 'monthly':
        rebalance_months = list(range(len(months)))
    elif frequency == 'quarterly':
        # Rebalance every 3 months
        rebalance_months = list(range(0, len(months), 3))
    elif frequency == 'semi-annual':
        # Rebalance every 6 months
        rebalance_months = list(range(0, len(months), 6))
    
    return months, rebalance_months

months_df, rebalance_indices = get_rebalancing_periods(pred, REBALANCE_FREQUENCY)
print(f"✓ Rebalancing schedule created")
print(f"   Total months: {len(months_df)}")
print(f"   Rebalancing months: {len(rebalance_indices)}")
print(f"   Rebalancing frequency: Every {len(months_df)/len(rebalance_indices):.1f} months on average")

# STEP 4: PORTFOLIO CONSTRUCTION FUNCTIONS
print("\n4. SETTING UP PORTFOLIO CONSTRUCTION...")
print("-" * 40)

def construct_eps_portfolio_allocation(data, model_col, long_n, short_n, strategy_name, verbose=False):
    """Construct portfolio based on EPS predictions with specific long/short allocation"""
    
    portfolio_data = []
    current_long_holdings = set()
    current_short_holdings = set()
    
    months = data.groupby(['year', 'month'])
    month_list = sorted(months.groups.keys())
    
    for i, (year, month) in enumerate(month_list):
        month_data = months.get_group((year, month))
        
        # Check if this is a rebalancing month
        is_rebalance_month = i in rebalance_indices
        
        if is_rebalance_month or i == 0:  # Always rebalance on first month
            # Sort by predicted EPS (high to low)
            month_data_sorted = month_data.sort_values(model_col, ascending=False)
            
            # Select holdings based on allocation
            if long_n > 0:
                new_long_holdings = set(month_data_sorted.head(long_n)['permno'])
            else:
                new_long_holdings = set()
                
            if short_n > 0:
                new_short_holdings = set(month_data_sorted.tail(short_n)['permno'])
            else:
                new_short_holdings = set()
            
            current_long_holdings = new_long_holdings
            current_short_holdings = new_short_holdings
            
            if verbose and i < 5:  # Only print first few for brevity
                print(f"   {strategy_name} - {year}-{month:02d}: {len(current_long_holdings)} long + {len(current_short_holdings)} short")
        
        # Calculate returns for current holdings
        long_stocks = month_data[month_data['permno'].isin(current_long_holdings)]
        short_stocks = month_data[month_data['permno'].isin(current_short_holdings)]
        
        # Calculate weighted returns
        if long_n > 0 and len(long_stocks) > 0:
            long_return = long_stocks['stock_exret'].mean()
            long_weight = long_n / (long_n + short_n) if (long_n + short_n) > 0 else 0
        else:
            long_return = 0
            long_weight = 0
            
        if short_n > 0 and len(short_stocks) > 0:
            short_return = short_stocks['stock_exret'].mean()
            short_weight = short_n / (long_n + short_n) if (long_n + short_n) > 0 else 0
        else:
            short_return = 0
            short_weight = 0
        
        # Portfolio return: weighted long return minus weighted short return
        if long_n > 0 and short_n > 0:
            # Traditional long-short: long positions - short positions
            strategy_return = long_return - short_return
        elif long_n > 0 and short_n == 0:
            # Long-only strategy
            strategy_return = long_return
        elif long_n == 0 and short_n > 0:
            # Short-only strategy (negative of short return)
            strategy_return = -short_return
        else:
            strategy_return = 0
        
        portfolio_data.append({
            'year': year,
            'month': month,
            'date': pd.to_datetime(f"{year}-{month:02d}-01"),
            'long_portfolio': long_return,
            'short_portfolio': short_return,
            'strategy_return': strategy_return,
            'long_holdings_count': len(long_stocks),
            'short_holdings_count': len(short_stocks),
            'total_holdings': len(long_stocks) + len(short_stocks),
            'rebalanced': is_rebalance_month,
            'strategy_name': strategy_name
        })
    
    return pd.DataFrame(portfolio_data)

def calculate_performance_metrics(returns, rf_rate, market_returns, strategy_name="Strategy"):
    """Calculate comprehensive performance metrics"""
    
    # Basic statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Annualized metrics
    annual_return = mean_return * 12
    annual_volatility = std_return * np.sqrt(12)
    
    # Sharpe ratio
    excess_returns = returns - rf_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12) if excess_returns.std() > 0 else 0
    
    # Information ratio (vs market)
    active_returns = returns - market_returns
    info_ratio = active_returns.mean() / active_returns.std() * np.sqrt(12) if active_returns.std() > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # VaR and CVaR (5% level)
    var_5 = returns.quantile(0.05)
    cvar_5 = returns[returns <= var_5].mean()
    
    # Beta vs market
    if len(market_returns) == len(returns):
        beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
        alpha = mean_return - beta * market_returns.mean()
        annual_alpha = alpha * 12
    else:
        beta = np.nan
        alpha = np.nan
        annual_alpha = np.nan
    
    # Tracking error
    tracking_error = active_returns.std() * np.sqrt(12)
    
    return {
        'strategy_name': strategy_name,
        'mean_monthly_return': mean_return,
        'monthly_volatility': std_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'information_ratio': info_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_5': var_5,
        'cvar_5': cvar_5,
        'beta': beta,
        'alpha': alpha,
        'annual_alpha': annual_alpha,
        'tracking_error': tracking_error,
        'total_months': len(returns)
    }

# STEP 5: RUN MULTI-ALLOCATION BACKTESTING
print("\n5. RUNNING MULTI-ALLOCATION BACKTESTING...")
print("-" * 40)

all_results = []
all_portfolio_returns = []

print(f"Testing {len(ALLOCATION_STRATEGIES)} different allocation strategies...")

for i, strategy in enumerate(ALLOCATION_STRATEGIES):
    long_n = strategy['long']
    short_n = strategy['short']
    strategy_name = strategy['name']
    
    print(f"\n📊 Strategy {i+1}/{len(ALLOCATION_STRATEGIES)}: {strategy_name}")
    print(f"   Long: {long_n}, Short: {short_n}")
    
    # Construct portfolio for this allocation
    portfolio_returns = construct_eps_portfolio_allocation(
        pred, model, long_n, short_n, strategy_name, verbose=(i==0)
    )
    
    # Merge with market data
    portfolio_returns['date'] = pd.to_datetime(portfolio_returns[['year', 'month']].assign(day=1))
    portfolio_with_market = portfolio_returns.merge(mkt_data[['date', 'sp_ret', 'rf']], on='date', how='left')
    portfolio_with_market = portfolio_with_market.rename(columns={'sp_ret': 'mkt_exret'})
    
    # Calculate performance metrics
    strategy_metrics = calculate_performance_metrics(
        portfolio_with_market['strategy_return'], 
        portfolio_with_market['rf'], 
        portfolio_with_market['mkt_exret'],
        strategy_name
    )
    
    # Add allocation info
    strategy_metrics['long_allocation'] = long_n
    strategy_metrics['short_allocation'] = short_n
    strategy_metrics['long_percentage'] = long_n / (long_n + short_n) * 100 if (long_n + short_n) > 0 else 0
    strategy_metrics['short_percentage'] = short_n / (long_n + short_n) * 100 if (long_n + short_n) > 0 else 0
    
    all_results.append(strategy_metrics)
    all_portfolio_returns.append(portfolio_with_market)
    
    print(f"   ✓ Annual Return: {strategy_metrics['annual_return']:.2%}")
    print(f"   ✓ Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.3f}")
    print(f"   ✓ Max Drawdown: {strategy_metrics['max_drawdown']:.2%}")

# STEP 6: ANALYZE RESULTS
print("\n6. ANALYZING ALLOCATION STRATEGY RESULTS...")
print("-" * 40)

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# Sort by Sharpe ratio (descending)
results_df = results_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)

print(f"\n🏆 ALLOCATION STRATEGY RANKINGS (by Sharpe Ratio):")
print("="*80)
print(f"{'Rank':<4} {'Strategy':<20} {'Long%':<6} {'Short%':<7} {'Ann.Ret':<8} {'Sharpe':<7} {'MaxDD':<8} {'Info.Ratio':<10}")
print("-"*80)

for i, row in results_df.iterrows():
    rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
    print(f"{rank_emoji:<4} {row['strategy_name']:<20} {row['long_percentage']:<6.0f} {row['short_percentage']:<7.0f} "
          f"{row['annual_return']:<8.2%} {row['sharpe_ratio']:<7.3f} {row['max_drawdown']:<8.2%} {row['information_ratio']:<10.3f}")

# Find best strategies by different metrics
best_sharpe = results_df.iloc[0]
best_return = results_df.loc[results_df['annual_return'].idxmax()]
best_drawdown = results_df.loc[results_df['max_drawdown'].idxmax()]  # Least negative
best_info_ratio = results_df.loc[results_df['information_ratio'].idxmax()]

print(f"\n📈 BEST STRATEGIES BY METRIC:")
print(f"   Best Sharpe Ratio: {best_sharpe['strategy_name']} ({best_sharpe['sharpe_ratio']:.3f})")
print(f"   Best Annual Return: {best_return['strategy_name']} ({best_return['annual_return']:.2%})")
print(f"   Best Max Drawdown: {best_drawdown['strategy_name']} ({best_drawdown['max_drawdown']:.2%})")
print(f"   Best Info Ratio: {best_info_ratio['strategy_name']} ({best_info_ratio['information_ratio']:.3f})")

# STEP 7: SAVE RESULTS
print("\n7. SAVING RESULTS...")
print("-" * 40)

# Create eps_results directory if it doesn't exist
os.makedirs("eps_results", exist_ok=True)

# Save detailed results
results_df.to_csv('eps_results/eps_multi_allocation_results.csv', index=False)
print("✓ Multi-allocation results saved to: eps_results/eps_multi_allocation_results.csv")

# Save all portfolio returns
all_returns_df = pd.concat(all_portfolio_returns, ignore_index=True)
all_returns_df.to_csv('eps_results/eps_multi_allocation_portfolio_returns.csv', index=False)
print("✓ All portfolio returns saved to: eps_results/eps_multi_allocation_portfolio_returns.csv")

# STEP 8: VISUALIZATION
print("\n8. CREATING VISUALIZATIONS...")
print("-" * 40)

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Sharpe Ratio vs Long Allocation
ax1.scatter(results_df['long_percentage'], results_df['sharpe_ratio'], 
           c=results_df['annual_return'], cmap='RdYlGn', s=100, alpha=0.7)
ax1.set_xlabel('Long Allocation (%)')
ax1.set_ylabel('Sharpe Ratio')
ax1.set_title('Sharpe Ratio vs Long Allocation')
ax1.grid(True, alpha=0.3)
# Add colorbar
cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
cbar1.set_label('Annual Return')

# Plot 2: Annual Return vs Long Allocation
ax2.scatter(results_df['long_percentage'], results_df['annual_return'], 
           c=results_df['max_drawdown'], cmap='RdYlGn_r', s=100, alpha=0.7)
ax2.set_xlabel('Long Allocation (%)')
ax2.set_ylabel('Annual Return')
ax2.set_title('Annual Return vs Long Allocation')
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
# Add colorbar
cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
cbar2.set_label('Max Drawdown')

# Plot 3: Risk-Return Scatter
ax3.scatter(results_df['annual_volatility'], results_df['annual_return'], 
           c=results_df['long_percentage'], cmap='viridis', s=100, alpha=0.7)
ax3.set_xlabel('Annual Volatility')
ax3.set_ylabel('Annual Return')
ax3.set_title('Risk-Return Profile by Allocation')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
# Add colorbar
cbar3 = plt.colorbar(ax3.collections[0], ax=ax3)
cbar3.set_label('Long Allocation (%)')

# Plot 4: Cumulative Returns for Top 3 Strategies
top_3_strategies = results_df.head(3)['strategy_name'].tolist()
for strategy_name in top_3_strategies:
    strategy_returns = all_returns_df[all_returns_df['strategy_name'] == strategy_name]
    cumulative_returns = (1 + strategy_returns['strategy_return']).cumprod()
    ax4.plot(strategy_returns['date'], cumulative_returns, label=strategy_name, linewidth=2)

# Add market benchmark
market_returns = all_returns_df[all_returns_df['strategy_name'] == top_3_strategies[0]]  # Use first strategy's dates
market_cumulative = (1 + market_returns['mkt_exret']).cumprod()
ax4.plot(market_returns['date'], market_cumulative, label='Market Benchmark', 
         linewidth=2, linestyle='--', alpha=0.7, color='black')

ax4.set_xlabel('Date')
ax4.set_ylabel('Cumulative Return')
ax4.set_title('Cumulative Returns - Top 3 Allocation Strategies')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eps_results/eps_multi_allocation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Multi-allocation analysis charts saved to: eps_results/eps_multi_allocation_analysis.png")

# STEP 9: SUMMARY AND RECOMMENDATIONS
print("\n" + "="*80)
print("EPS-BASED MULTI-ALLOCATION BACKTEST SUMMARY")
print("="*80)

print(f"\n📊 ANALYSIS OVERVIEW:")
print(f"   Strategies Tested: {len(ALLOCATION_STRATEGIES)}")
print(f"   Model: {best_model_info.get('model_type', 'Ridge').title()} with {best_model_info.get('n_features', 75)} features")
print(f"   Period: {all_returns_df['date'].min().strftime('%Y-%m')} to {all_returns_df['date'].max().strftime('%Y-%m')}")
print(f"   Total Holdings: {TOTAL_HOLDINGS}")
print(f"   Rebalancing: {REBALANCE_FREQUENCY.title()}")

print(f"\n🏆 OPTIMAL ALLOCATION STRATEGY:")
print(f"   Best Strategy: {best_sharpe['strategy_name']}")
print(f"   Long Allocation: {best_sharpe['long_allocation']} stocks ({best_sharpe['long_percentage']:.0f}%)")
print(f"   Short Allocation: {best_sharpe['short_allocation']} stocks ({best_sharpe['short_percentage']:.0f}%)")
print(f"   Annual Return: {best_sharpe['annual_return']:.2%}")
print(f"   Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
print(f"   Max Drawdown: {best_sharpe['max_drawdown']:.2%}")
print(f"   Information Ratio: {best_sharpe['information_ratio']:.3f}")

print(f"\n📈 KEY INSIGHTS:")
# Calculate some insights
long_only = results_df[results_df['strategy_name'] == 'Long-Only'].iloc[0]
short_only = results_df[results_df['strategy_name'] == 'Short-Only'].iloc[0]
balanced = results_df[results_df['strategy_name'] == '50/50 Long-Short'].iloc[0]

print(f"   Long-Only Performance: {long_only['annual_return']:.2%} return, {long_only['sharpe_ratio']:.3f} Sharpe")
print(f"   Short-Only Performance: {short_only['annual_return']:.2%} return, {short_only['sharpe_ratio']:.3f} Sharpe")
print(f"   Balanced (50/50) Performance: {balanced['annual_return']:.2%} return, {balanced['sharpe_ratio']:.3f} Sharpe")

# Find optimal range
top_5_strategies = results_df.head(5)
optimal_long_range = (top_5_strategies['long_percentage'].min(), top_5_strategies['long_percentage'].max())
print(f"   Optimal Long Allocation Range: {optimal_long_range[0]:.0f}% - {optimal_long_range[1]:.0f}%")

print(f"\n📁 OUTPUT FILES:")
print(f"   • eps_results/eps_multi_allocation_results.csv - Detailed strategy comparison")
print(f"   • eps_results/eps_multi_allocation_portfolio_returns.csv - All portfolio returns")
print(f"   • eps_results/eps_multi_allocation_analysis.png - Comprehensive analysis charts")

if best_sharpe['sharpe_ratio'] > 0.5:
    print(f"\n✅ EXCELLENT RESULT! Optimal allocation found with Sharpe ratio > 0.5")
elif best_sharpe['sharpe_ratio'] > 0.3:
    print(f"\n✅ GOOD RESULT! Reasonable allocation found with Sharpe ratio > 0.3")
else:
    print(f"\n⚠️  MODERATE RESULT. Consider further strategy refinements.")

print("\n" + "="*80)
print("MULTI-ALLOCATION BACKTEST COMPLETED SUCCESSFULLY!")
print("="*80) 