import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("EPS-BASED PORTFOLIO BACKTESTING ANALYSIS")
print("="*80)

# CONFIGURATION PARAMETERS
REBALANCE_FREQUENCY = 'monthly'  # Options: 'monthly', 'quarterly', 'semi-annual'
MIN_HOLDINGS = 50
MAX_HOLDINGS = 100
LONG_HOLDINGS = 50  # Number of long positions (high EPS growth)
SHORT_HOLDINGS = 50  # Number of short positions (low EPS growth)

print(f"STRATEGY CONFIGURATION:")
print(f"   Strategy: Long high EPS growth, Short low EPS growth")
print(f"   Rebalancing Frequency: {REBALANCE_FREQUENCY}")
print(f"   Long Holdings: {LONG_HOLDINGS}")
print(f"   Short Holdings: {SHORT_HOLDINGS}")
print(f"   Total Holdings: {LONG_HOLDINGS + SHORT_HOLDINGS}")
print(f"   Holdings Range Check: {MIN_HOLDINGS} ≤ {LONG_HOLDINGS + SHORT_HOLDINGS} ≤ {MAX_HOLDINGS}")

# Validate holdings constraint
total_holdings = LONG_HOLDINGS + SHORT_HOLDINGS
if not (MIN_HOLDINGS <= total_holdings <= MAX_HOLDINGS):
    print(f"❌ ERROR: Total holdings ({total_holdings}) must be between {MIN_HOLDINGS} and {MAX_HOLDINGS}")
    exit(1)

print("✅ Holdings constraint satisfied")

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

# STEP 4: EPS-BASED PORTFOLIO CONSTRUCTION
print("\n4. CONSTRUCTING EPS-BASED PORTFOLIOS...")
print("-" * 40)

def construct_eps_portfolio_holdings(data, model_col, long_n, short_n):
    """Construct portfolio based on EPS predictions - Long high EPS growth, Short low EPS growth"""
    
    portfolio_data = []
    detailed_holdings = []  # Store detailed holdings with tickers and company names
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
            
            # Select top N for long (highest predicted EPS), bottom N for short (lowest predicted EPS)
            new_long_holdings = set(month_data_sorted.head(long_n)['permno'])
            new_short_holdings = set(month_data_sorted.tail(short_n)['permno'])
            
            current_long_holdings = new_long_holdings
            current_short_holdings = new_short_holdings
            
            print(f"   Rebalanced {year}-{month:02d}: {len(current_long_holdings)} long + {len(current_short_holdings)} short = {len(current_long_holdings) + len(current_short_holdings)} total")
            
            # Store detailed holdings for this rebalancing month
            # Check if ticker/company info is available
            base_columns = ['permno', model_col, 'eps_actual', 'stock_exret']
            optional_columns = []
            
            if 'stock_ticker' in month_data_sorted.columns:
                optional_columns.append('stock_ticker')
            if 'comp_name' in month_data_sorted.columns:
                optional_columns.append('comp_name')
            
            columns_to_select = base_columns + optional_columns
            
            long_details = month_data_sorted.head(long_n)[columns_to_select].copy()
            long_details['position_type'] = 'LONG'
            long_details['year'] = year
            long_details['month'] = month
            long_details['date'] = pd.to_datetime(f"{year}-{month:02d}-01")
            
            short_details = month_data_sorted.tail(short_n)[columns_to_select].copy()
            short_details['position_type'] = 'SHORT'
            short_details['year'] = year
            short_details['month'] = month
            short_details['date'] = pd.to_datetime(f"{year}-{month:02d}-01")
            
            detailed_holdings.append(long_details)
            detailed_holdings.append(short_details)
        
        # Calculate returns for current holdings
        long_stocks = month_data[month_data['permno'].isin(current_long_holdings)]
        short_stocks = month_data[month_data['permno'].isin(current_short_holdings)]
        
        # Equal-weighted returns
        long_return = long_stocks['stock_exret'].mean() if len(long_stocks) > 0 else 0
        short_return = short_stocks['stock_exret'].mean() if len(short_stocks) > 0 else 0
        strategy_return = long_return - short_return
        
        portfolio_data.append({
            'year': year,
            'month': month,
            'date': pd.to_datetime(f"{year}-{month:02d}-01"),
            'long_portfolio': long_return,
            'short_portfolio': short_return,
            'long_short_return': strategy_return,
            'long_holdings_count': len(long_stocks),
            'short_holdings_count': len(short_stocks),
            'total_holdings': len(long_stocks) + len(short_stocks),
            'rebalanced': is_rebalance_month
        })
    
    # Save detailed holdings to CSV
    if detailed_holdings:
        detailed_df = pd.concat(detailed_holdings, ignore_index=True)
        detailed_df = detailed_df.rename(columns={model_col: 'predicted_eps'})
        
        # Create eps_results directory if it doesn't exist
        os.makedirs("eps_results", exist_ok=True)
        
        detailed_df.to_csv('eps_results/eps_portfolio_detailed_holdings.csv', index=False)
        print(f"✓ Detailed EPS holdings saved to: eps_results/eps_portfolio_detailed_holdings.csv")
        print(f"   Contains {len(detailed_df)} holdings across {detailed_df['date'].nunique()} rebalancing dates")
    
    return pd.DataFrame(portfolio_data)

# Run portfolio construction
portfolio_returns = construct_eps_portfolio_holdings(pred, model, LONG_HOLDINGS, SHORT_HOLDINGS)

print(f"\n✓ Portfolio construction completed!")
print(f"   Total months: {len(portfolio_returns)}")
print(f"   Average long holdings: {portfolio_returns['long_holdings_count'].mean():.1f}")
print(f"   Average short holdings: {portfolio_returns['short_holdings_count'].mean():.1f}")
print(f"   Average total holdings: {portfolio_returns['total_holdings'].mean():.1f}")

# STEP 5: PERFORMANCE METRICS CALCULATION
print("\n5. CALCULATING PERFORMANCE METRICS...")
print("-" * 40)

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
    
    # Maximum one-month loss
    max_monthly_loss = returns.min()
    
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
        'total_months': len(returns),
        'max_monthly_loss': max_monthly_loss
    }

# Merge with market data
portfolio_returns['date'] = pd.to_datetime(portfolio_returns[['year', 'month']].assign(day=1))
portfolio_with_market = portfolio_returns.merge(mkt_data[['date', 'sp_ret', 'rf']], on='date', how='left')

# Rename sp_ret to mkt_exret for consistency
portfolio_with_market = portfolio_with_market.rename(columns={'sp_ret': 'mkt_exret'})

# Calculate metrics for different strategies
long_short_metrics = calculate_performance_metrics(
    portfolio_with_market['long_short_return'], 
    portfolio_with_market['rf'], 
    portfolio_with_market['mkt_exret'],
    "EPS Long-Short Strategy"
)

long_only_metrics = calculate_performance_metrics(
    portfolio_with_market['long_portfolio'], 
    portfolio_with_market['rf'], 
    portfolio_with_market['mkt_exret'],
    "EPS Long-Only Strategy"
)

short_only_metrics = calculate_performance_metrics(
    portfolio_with_market['short_portfolio'], 
    portfolio_with_market['rf'], 
    portfolio_with_market['mkt_exret'],
    "EPS Short-Only Strategy"
)

market_metrics = calculate_performance_metrics(
    portfolio_with_market['mkt_exret'], 
    portfolio_with_market['rf'], 
    portfolio_with_market['mkt_exret'],
    "Market Benchmark"
)

# STEP 6: TURNOVER CALCULATION
print("\n6. CALCULATING TURNOVER...")
print("-" * 40)

def calculate_corrected_turnover(data, model_col, long_n, short_n, rebalance_indices):
    """Calculate portfolio turnover correctly"""
    
    months = data.groupby(['year', 'month'])
    month_list = sorted(months.groups.keys())
    
    turnovers = []
    prev_long_holdings = set()
    prev_short_holdings = set()
    
    for i, (year, month) in enumerate(month_list):
        month_data = months.get_group((year, month))
        
        # Check if this is a rebalancing month
        is_rebalance_month = i in rebalance_indices
        
        if is_rebalance_month:
            # Sort by predicted EPS
            month_data_sorted = month_data.sort_values(model_col, ascending=False)
            
            # Current holdings
            current_long_holdings = set(month_data_sorted.head(long_n)['permno'])
            current_short_holdings = set(month_data_sorted.tail(short_n)['permno'])
            
            if i > 0:  # Skip first month (no previous holdings to compare)
                # Calculate turnover
                long_unchanged = len(current_long_holdings.intersection(prev_long_holdings))
                short_unchanged = len(current_short_holdings.intersection(prev_short_holdings))
                
                long_turnover = (long_n - long_unchanged) / long_n if long_n > 0 else 0
                short_turnover = (short_n - short_unchanged) / short_n if short_n > 0 else 0
                
                # Average turnover
                avg_turnover = (long_turnover + short_turnover) / 2
                
                turnovers.append({
                    'year': year,
                    'month': month,
                    'long_turnover': long_turnover,
                    'short_turnover': short_turnover,
                    'avg_turnover': avg_turnover
                })
            
            # Update previous holdings
            prev_long_holdings = current_long_holdings.copy()
            prev_short_holdings = current_short_holdings.copy()
    
    return pd.DataFrame(turnovers)

turnover_df = calculate_corrected_turnover(pred, model, LONG_HOLDINGS, SHORT_HOLDINGS, rebalance_indices)

if len(turnover_df) > 0:
    avg_turnover = turnover_df['avg_turnover'].mean()
    print(f"✓ Average monthly turnover: {avg_turnover:.1%}")
    print(f"   Long portfolio turnover: {turnover_df['long_turnover'].mean():.1%}")
    print(f"   Short portfolio turnover: {turnover_df['short_turnover'].mean():.1%}")
else:
    avg_turnover = 0
    print("⚠ No turnover data available")

# STEP 7: RESULTS DISPLAY
print("\n7. PERFORMANCE RESULTS")
print("="*80)

def print_metrics(metrics):
    """Print performance metrics in a formatted way"""
    print(f"\n{metrics['strategy_name'].upper()}")
    print("-" * len(metrics['strategy_name']))
    print(f"Annual Return:        {metrics['annual_return']:>8.2%}")
    print(f"Annual Volatility:    {metrics['annual_volatility']:>8.2%}")
    print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:>8.2f}")
    print(f"Information Ratio:    {metrics['information_ratio']:>8.2f}")
    print(f"Max Drawdown:         {metrics['max_drawdown']:>8.2%}")
    print(f"Calmar Ratio:         {metrics['calmar_ratio']:>8.2f}")
    print(f"Win Rate:             {metrics['win_rate']:>8.2%}")
    print(f"Beta:                 {metrics['beta']:>8.2f}")
    print(f"Annual Alpha:         {metrics['annual_alpha']:>8.2%}")
    print(f"Tracking Error:       {metrics['tracking_error']:>8.2%}")
    print(f"Skewness:             {metrics['skewness']:>8.2f}")
    print(f"Kurtosis:             {metrics['kurtosis']:>8.2f}")
    print(f"VaR (5%):             {metrics['var_5']:>8.2%}")
    print(f"CVaR (5%):            {metrics['cvar_5']:>8.2%}")
    print(f"Max Monthly Loss:      {metrics['max_monthly_loss']:>8.2%}")

# Print all metrics
print_metrics(long_short_metrics)
print_metrics(long_only_metrics)
print_metrics(short_only_metrics)
print_metrics(market_metrics)

print(f"\nTURNOVER ANALYSIS")
print("-" * 20)
print(f"Average Monthly Turnover: {avg_turnover:.1%}")

# STEP 8: SAVE RESULTS
print("\n8. SAVING RESULTS...")
print("-" * 40)

# Create eps_results directory if it doesn't exist
os.makedirs("eps_results", exist_ok=True)

# Save portfolio returns
portfolio_with_market.to_csv('eps_results/eps_portfolio_returns.csv', index=False)
print("✓ Portfolio returns saved to: eps_results/eps_portfolio_returns.csv")

# Save turnover data
if len(turnover_df) > 0:
    turnover_df.to_csv('eps_results/eps_portfolio_turnover.csv', index=False)
    print("✓ Turnover data saved to: eps_results/eps_portfolio_turnover.csv")

# Save performance metrics
metrics_df = pd.DataFrame([long_short_metrics, long_only_metrics, short_only_metrics, market_metrics])
metrics_df.to_csv('eps_results/eps_portfolio_performance_metrics.csv', index=False)
print("✓ Performance metrics saved to: eps_results/eps_portfolio_performance_metrics.csv")

# STEP 9: VISUALIZATION
print("\n9. CREATING VISUALIZATIONS...")
print("-" * 40)

# Calculate cumulative returns
portfolio_with_market['long_short_cumret'] = (1 + portfolio_with_market['long_short_return']).cumprod()
portfolio_with_market['long_cumret'] = (1 + portfolio_with_market['long_portfolio']).cumprod()
portfolio_with_market['market_cumret'] = (1 + portfolio_with_market['mkt_exret']).cumprod()

# Create plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Cumulative Returns
ax1.plot(portfolio_with_market['date'], portfolio_with_market['long_short_cumret'], 
         label='EPS Long-Short', linewidth=2, color='blue')
ax1.plot(portfolio_with_market['date'], portfolio_with_market['long_cumret'], 
         label='EPS Long-Only', linewidth=2, color='green')
ax1.plot(portfolio_with_market['date'], portfolio_with_market['market_cumret'], 
         label='Market', linewidth=2, color='red', alpha=0.7)
ax1.set_title('Cumulative Returns - EPS-Based Strategy')
ax1.set_ylabel('Cumulative Return')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Rolling Sharpe Ratio (12-month)
rolling_window = 12
portfolio_with_market['rolling_sharpe_ls'] = (
    (portfolio_with_market['long_short_return'] - portfolio_with_market['rf'])
    .rolling(rolling_window).mean() / 
    (portfolio_with_market['long_short_return'] - portfolio_with_market['rf'])
    .rolling(rolling_window).std() * np.sqrt(12)
)

ax2.plot(portfolio_with_market['date'], portfolio_with_market['rolling_sharpe_ls'], 
         linewidth=2, color='blue')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_title(f'Rolling {rolling_window}-Month Sharpe Ratio')
ax2.set_ylabel('Sharpe Ratio')
ax2.grid(True, alpha=0.3)

# Plot 3: Monthly Returns Distribution
ax3.hist(portfolio_with_market['long_short_return'], bins=30, alpha=0.7, 
         color='blue', label='EPS Long-Short')
ax3.hist(portfolio_with_market['mkt_exret'], bins=30, alpha=0.5, 
         color='red', label='Market')
ax3.set_title('Monthly Returns Distribution')
ax3.set_xlabel('Monthly Return')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Drawdown
portfolio_with_market['rolling_max'] = portfolio_with_market['long_short_cumret'].expanding().max()
portfolio_with_market['drawdown'] = (portfolio_with_market['long_short_cumret'] - 
                                    portfolio_with_market['rolling_max']) / portfolio_with_market['rolling_max']

ax4.fill_between(portfolio_with_market['date'], portfolio_with_market['drawdown'], 0, 
                 alpha=0.7, color='red')
ax4.set_title('Drawdown - EPS Long-Short Strategy')
ax4.set_ylabel('Drawdown')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eps_results/eps_portfolio_performance.png', dpi=300, bbox_inches='tight')
print("✓ Performance charts saved to: eps_results/eps_portfolio_performance.png")

# STEP 10: SUMMARY
print("\n" + "="*80)
print("EPS-BASED PORTFOLIO BACKTEST SUMMARY")
print("="*80)

print(f"\n📊 STRATEGY OVERVIEW:")
print(f"   Strategy: Long high EPS growth stocks, Short low EPS growth stocks")
print(f"   Model: {best_model_info.get('model_type', 'Ridge').title()} with {best_model_info.get('n_features', 75)} features")
print(f"   EPS Model R²: {best_model_info.get('overall_r2', 0.201381):.4f}")
print(f"   Period: {portfolio_with_market['date'].min().strftime('%Y-%m')} to {portfolio_with_market['date'].max().strftime('%Y-%m')}")
print(f"   Total Months: {len(portfolio_with_market)}")
print(f"   Rebalancing: {REBALANCE_FREQUENCY.title()}")

print(f"\n🎯 KEY PERFORMANCE METRICS:")
print(f"   Long-Short Annual Return: {long_short_metrics['annual_return']:>8.2%}")
print(f"   Long-Short Sharpe Ratio:  {long_short_metrics['sharpe_ratio']:>8.2f}")
print(f"   Long-Short Max Drawdown:  {long_short_metrics['max_drawdown']:>8.2%}")
print(f"   Long-Short Alpha:         {long_short_metrics['annual_alpha']:>8.2%}")
print(f"   Average Turnover:         {avg_turnover:>8.1%}")

print(f"\n📈 COMPARISON TO MARKET:")
print(f"   Strategy vs Market Return: {long_short_metrics['annual_return'] - market_metrics['annual_return']:>+8.2%}")
print(f"   Strategy vs Market Sharpe: {long_short_metrics['sharpe_ratio'] - market_metrics['sharpe_ratio']:>+8.2f}")
print(f"   Information Ratio:         {long_short_metrics['information_ratio']:>8.2f}")

print(f"\n💼 PORTFOLIO CHARACTERISTICS:")
print(f"   Average Long Holdings:     {portfolio_returns['long_holdings_count'].mean():>8.1f}")
print(f"   Average Short Holdings:    {portfolio_returns['short_holdings_count'].mean():>8.1f}")
print(f"   Average Total Holdings:    {portfolio_returns['total_holdings'].mean():>8.1f}")

print(f"\n📁 OUTPUT FILES:")
print(f"   • eps_results/eps_portfolio_returns.csv - Monthly portfolio returns")
print(f"   • eps_results/eps_portfolio_detailed_holdings.csv - Detailed holdings by rebalancing date")
print(f"   • eps_results/eps_portfolio_performance_metrics.csv - Performance metrics")
print(f"   • eps_results/eps_portfolio_turnover.csv - Turnover analysis")
print(f"   • eps_results/eps_portfolio_performance.png - Performance charts")

if long_short_metrics['sharpe_ratio'] > 1.0:
    print(f"\n🎉 EXCELLENT PERFORMANCE! Sharpe ratio > 1.0")
elif long_short_metrics['sharpe_ratio'] > 0.5:
    print(f"\n✅ GOOD PERFORMANCE! Sharpe ratio > 0.5")
else:
    print(f"\n⚠️  MODERATE PERFORMANCE. Consider strategy refinements.")

print("\n" + "="*80)
print("EPS-BASED BACKTEST COMPLETED SUCCESSFULLY!")
print("="*80) 