import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("COMPLETE PORTFOLIO BACKTESTING ANALYSIS - 50-100 HOLDINGS")
print("="*80)

# CONFIGURATION PARAMETERS
REBALANCE_FREQUENCY = 'monthly'  # Options: 'monthly', 'quarterly', 'semi-annual'
MIN_HOLDINGS = 50
MAX_HOLDINGS = 100
LONG_HOLDINGS = 50  # Number of long positions
SHORT_HOLDINGS = 50  # Number of short positions

print(f"STRATEGY CONFIGURATION:")
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

# STEP 1: LOAD BEST MODEL PREDICTIONS
print("\n1. LOADING PREDICTIONS...")
print("-" * 40)

if os.path.exists('best_model_info.json'):
    with open('best_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    
    print(f"✓ Best model info found!")
    print(f"   Model: {best_model_info['model_type'].title()}")
    print(f"   Features: {best_model_info['n_features']}")
    print(f"   Overall R²: {best_model_info['overall_r2']:.6f}")
    
    pred_path = best_model_info['predictions_file']
    model = best_model_info['model_type']
    
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        print(f"✓ Predictions loaded from {pred_path}")
        
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
        print(f"✗ ERROR: Predictions file {pred_path} not found!")
        exit(1)
        
else:
    print("⚠ WARNING: best_model_info.json not found! Using fallback...")
    pred_path = "stock_predictions_ridge.csv"  
    model = "ridge"
    
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        print(f"✓ Fallback predictions loaded from {pred_path}")
        
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
        print(f"✗ ERROR: No predictions file found!")
        exit(1)

print(f"Data shape: {pred.shape}")
print(f"Date range: {pred['date'].min().strftime('%Y-%m')} to {pred['date'].max().strftime('%Y-%m')}")

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

# STEP 4: PORTFOLIO CONSTRUCTION WITH 50-100 HOLDINGS
print("\n4. CONSTRUCTING PORTFOLIOS WITH 50-100 HOLDINGS...")
print("-" * 40)

def construct_portfolio_holdings(data, model_col, long_n, short_n):
    """Construct portfolio with specified number of long and short holdings"""
    
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
            # Sort by predicted returns
            month_data_sorted = month_data.sort_values(model_col, ascending=False)
            
            # Select top N for long, bottom N for short
            new_long_holdings = set(month_data_sorted.head(long_n)['permno'])
            new_short_holdings = set(month_data_sorted.tail(short_n)['permno'])
            
            current_long_holdings = new_long_holdings
            current_short_holdings = new_short_holdings
            
            print(f"   Rebalanced {year}-{month:02d}: {len(current_long_holdings)} long + {len(current_short_holdings)} short = {len(current_long_holdings) + len(current_short_holdings)} total")
            
            # Store detailed holdings for this rebalancing month
            # Check if ticker/company info is available
            base_columns = ['permno', model_col, 'stock_exret']
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
        detailed_df = detailed_df.rename(columns={model_col: 'predicted_return'})
        detailed_df.to_csv('portfolio_detailed_holdings.csv', index=False)
        print(f"✓ Detailed holdings saved to: portfolio_detailed_holdings.csv")
        print(f"   Contains {len(detailed_df)} holdings across {detailed_df['date'].nunique()} rebalancing dates")
    
    return pd.DataFrame(portfolio_data)

# Construct portfolios
portfolio_returns_wide = construct_portfolio_holdings(pred, model, LONG_HOLDINGS, SHORT_HOLDINGS)

print(f"✓ Portfolio construction complete")
print(f"   Average long holdings: {portfolio_returns_wide['long_holdings_count'].mean():.1f}")
print(f"   Average short holdings: {portfolio_returns_wide['short_holdings_count'].mean():.1f}")
print(f"   Average total holdings: {portfolio_returns_wide['total_holdings'].mean():.1f}")
print(f"   Holdings range: {portfolio_returns_wide['total_holdings'].min()} to {portfolio_returns_wide['total_holdings'].max()}")
print(f"   Long portfolio average return: {portfolio_returns_wide['long_portfolio'].mean():.4f}")
print(f"   Short portfolio average return: {portfolio_returns_wide['short_portfolio'].mean():.4f}")
print(f"   Long-short strategy average return: {portfolio_returns_wide['long_short_return'].mean():.4f}")

# STEP 5: MERGE WITH MARKET DATA
print("\n5. MERGING WITH MARKET DATA...")
print("-" * 40)

# Merge portfolio returns with market data
results = pd.merge(portfolio_returns_wide, mkt_data, on=['year', 'month'], how='inner')
results = results.sort_values('date_x').reset_index(drop=True)
results = results.rename(columns={'date_x': 'date'})  # Use portfolio date column

print(f"✓ Merged data: {len(results)} months")
print(f"   Date range: {results['date'].min().strftime('%Y-%m')} to {results['date'].max().strftime('%Y-%m')}")

# STEP 6: CALCULATE PERFORMANCE METRICS
print("\n6. CALCULATING PERFORMANCE METRICS...")
print("-" * 40)

def calculate_performance_metrics(returns, rf_rate, market_returns, strategy_name="Strategy"):
    """Calculate comprehensive performance metrics"""
    
    # Basic statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Excess returns
    excess_returns = returns - rf_rate
    market_excess = market_returns - rf_rate
    
    # Sharpe Ratio
    sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Alpha and Beta (CAPM regression)
    if len(excess_returns) > 1 and market_excess.std() > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(market_excess, excess_returns)
        alpha = intercept
        beta = slope
        r_squared = r_value ** 2
    else:
        alpha, beta, r_squared = 0, 0, 0
    
    # Information Ratio (excess return vs tracking error)
    tracking_error = (excess_returns - market_excess).std()
    information_ratio = (excess_returns.mean() - market_excess.mean()) / tracking_error if tracking_error > 0 else 0
    
    # Cumulative returns for drawdown calculation
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown = drawdown.min()
    
    # Maximum one-month loss
    max_monthly_loss = returns.min()
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Annualized metrics (assuming monthly data)
    annualized_return = (1 + mean_return) ** 12 - 1
    annualized_volatility = std_return * np.sqrt(12)
    annualized_sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    return {
        'strategy_name': strategy_name,
        'total_months': len(returns),
        'mean_monthly_return': mean_return,
        'monthly_volatility': std_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'annualized_sharpe': annualized_sharpe,
        'alpha_monthly': alpha,
        'alpha_annualized': alpha * 12,
        'beta': beta,
        'r_squared': r_squared,
        'information_ratio': information_ratio,
        'max_drawdown': max_drawdown,
        'max_monthly_loss': max_monthly_loss,
        'win_rate': win_rate,
        'cumulative_return': cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
    }

# Calculate metrics for long-short strategy
ls_metrics = calculate_performance_metrics(
    results['long_short_return'], 
    results['rf'], 
    results['sp_ret'],
    "Long-Short Strategy"
)

# Calculate metrics for long portfolio only
long_metrics = calculate_performance_metrics(
    results['long_portfolio'], 
    results['rf'], 
    results['sp_ret'],
    "Long Portfolio Only"
)

# Calculate metrics for short portfolio only  
short_metrics = calculate_performance_metrics(
    results['short_portfolio'], 
    results['rf'], 
    results['sp_ret'],
    "Short Portfolio Only"
)

# Calculate metrics for market benchmark
market_metrics = calculate_performance_metrics(
    results['sp_ret'], 
    results['rf'], 
    results['sp_ret'],
    "S&P 500 Benchmark"
)

print("✓ Performance metrics calculated for all strategies")

# STEP 7: CALCULATE TURNOVER
print("\n7. CALCULATING TURNOVER...")
print("-" * 40)

def calculate_corrected_turnover(data, model_col, long_n, short_n, rebalance_indices):
    """Calculate turnover for the corrected strategy"""
    
    turnover_data = []
    months = data.groupby(['year', 'month'])
    month_list = sorted(months.groups.keys())
    
    prev_long_holdings = set()
    prev_short_holdings = set()
    
    for i, (year, month) in enumerate(month_list):
        if i == 0:
            continue  # Skip first month
            
        month_data = months.get_group((year, month))
        is_rebalance_month = i in rebalance_indices
        
        if is_rebalance_month:
            # Get new holdings
            month_data_sorted = month_data.sort_values(model_col, ascending=False)
            new_long_holdings = set(month_data_sorted.head(long_n)['permno'])
            new_short_holdings = set(month_data_sorted.tail(short_n)['permno'])
            
            # Calculate turnover
            long_unchanged = len(prev_long_holdings.intersection(new_long_holdings))
            short_unchanged = len(prev_short_holdings.intersection(new_short_holdings))
            
            long_turnover = 1 - (long_unchanged / long_n) if long_n > 0 else 0
            short_turnover = 1 - (short_unchanged / short_n) if short_n > 0 else 0
            
            turnover_data.append({
                'year': year,
                'month': month,
                'long_turnover': long_turnover,
                'short_turnover': short_turnover,
                'overall_turnover': (long_turnover + short_turnover) / 2,
                'rebalanced': True
            })
            
            prev_long_holdings = new_long_holdings
            prev_short_holdings = new_short_holdings
        else:
            # No rebalancing, turnover = 0
            turnover_data.append({
                'year': year,
                'month': month,
                'long_turnover': 0.0,
                'short_turnover': 0.0,
                'overall_turnover': 0.0,
                'rebalanced': False
            })
    
    return pd.DataFrame(turnover_data)

turnover_df = calculate_corrected_turnover(pred, model, LONG_HOLDINGS, SHORT_HOLDINGS, rebalance_indices)

# Calculate average turnover only for rebalancing months
rebalance_turnover = turnover_df[turnover_df['rebalanced']]
avg_turnover_when_rebalancing = rebalance_turnover['overall_turnover'].mean() if len(rebalance_turnover) > 0 else 0

# Calculate average monthly turnover (including non-rebalancing months)
avg_monthly_turnover = turnover_df['overall_turnover'].mean()

print(f"✓ Turnover calculated")
print(f"   Average turnover when rebalancing: {avg_turnover_when_rebalancing:.2%}")
print(f"   Average monthly turnover (including 0% months): {avg_monthly_turnover:.2%}")
print(f"   Number of rebalancing events: {len(rebalance_turnover)}")

# STEP 8: DISPLAY COMPREHENSIVE RESULTS
print("\n" + "="*80)
print("COMPREHENSIVE PORTFOLIO PERFORMANCE RESULTS")
print("="*80)

def print_metrics(metrics):
    """Print performance metrics in a formatted way"""
    print(f"\n📊 {metrics['strategy_name'].upper()}")
    print("-" * 50)
    print(f"Total Months: {metrics['total_months']}")
    print(f"Mean Monthly Return: {metrics['mean_monthly_return']:.4f} ({metrics['mean_monthly_return']*100:.2f}%)")
    print(f"Monthly Volatility: {metrics['monthly_volatility']:.4f} ({metrics['monthly_volatility']*100:.2f}%)")
    print(f"Annualized Return: {metrics['annualized_return']:.4f} ({metrics['annualized_return']*100:.2f}%)")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.4f} ({metrics['annualized_volatility']*100:.2f}%)")
    print(f"Cumulative Return: {metrics['cumulative_return']:.4f} ({metrics['cumulative_return']*100:.2f}%)")
    print(f"")
    print(f"📈 RISK-ADJUSTED METRICS:")
    print(f"Sharpe Ratio (Monthly): {metrics['sharpe_ratio']:.4f}")
    print(f"Sharpe Ratio (Annualized): {metrics['annualized_sharpe']:.4f}")
    print(f"Information Ratio: {metrics['information_ratio']:.4f}")
    print(f"")
    print(f"📊 CAPM REGRESSION RESULTS:")
    print(f"Alpha (Monthly): {metrics['alpha_monthly']:.4f} ({metrics['alpha_monthly']*100:.2f}%)")
    print(f"Alpha (Annualized): {metrics['alpha_annualized']:.4f} ({metrics['alpha_annualized']*100:.2f}%)")
    print(f"Beta: {metrics['beta']:.4f}")
    print(f"R-squared: {metrics['r_squared']:.4f}")
    print(f"")
    print(f"📉 RISK METRICS:")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
    print(f"Maximum Monthly Loss: {metrics['max_monthly_loss']:.4f} ({metrics['max_monthly_loss']*100:.2f}%)")
    print(f"Win Rate: {metrics['win_rate']:.4f} ({metrics['win_rate']*100:.2f}%)")

# Print all metrics
print_metrics(ls_metrics)
print_metrics(long_metrics)
print_metrics(short_metrics)
print_metrics(market_metrics)

# STEP 9: SUMMARY COMPARISON TABLE
print("\n" + "="*80)
print("STRATEGY COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame([ls_metrics, long_metrics, short_metrics, market_metrics])
comparison_cols = ['strategy_name', 'annualized_return', 'annualized_volatility', 'annualized_sharpe', 
                  'alpha_annualized', 'beta', 'max_drawdown', 'win_rate']
comparison_summary = comparison_df[comparison_cols].round(4)
print(comparison_summary.to_string(index=False))

# STEP 10: TURNOVER ANALYSIS
print(f"\n📊 TURNOVER ANALYSIS:")
print("-" * 30)
print(f"Average Monthly Turnover:")
print(f"  When Rebalancing: {avg_turnover_when_rebalancing:.2%}")
print(f"  Overall Monthly Average: {avg_monthly_turnover:.2%}")
print(f"  Rebalancing Events: {len(rebalance_turnover)} times")
print(f"\nTurnover Interpretation:")
print(f"  - High turnover (>50%) indicates frequent portfolio changes")
print(f"  - Low turnover (<20%) indicates stable portfolio composition")
print(f"  - Current turnover suggests {'high' if avg_turnover_when_rebalancing > 0.5 else 'moderate' if avg_turnover_when_rebalancing > 0.2 else 'low'} trading activity")

# STEP 10.5: STRATEGY COMPLIANCE CHECK
print(f"\n" + "="*80)
print("STRATEGY COMPLIANCE VERIFICATION")
print("="*80)

print(f"\n✅ HOLDINGS REQUIREMENTS:")
print(f"   Required: 50-100 total holdings")
print(f"   Actual: {LONG_HOLDINGS + SHORT_HOLDINGS} total holdings")
print(f"   Compliance: {'✅ PASS' if MIN_HOLDINGS <= (LONG_HOLDINGS + SHORT_HOLDINGS) <= MAX_HOLDINGS else '❌ FAIL'}")

print(f"\n✅ REBALANCING REQUIREMENTS:")
print(f"   Required: At least once every 6 months")
print(f"   Actual: {REBALANCE_FREQUENCY} rebalancing")
print(f"   Rebalancing events: {len(rebalance_turnover)} times over {len(results)} months")
print(f"   Average months between rebalancing: {len(results)/len(rebalance_turnover):.1f}")
print(f"   Compliance: {'✅ PASS' if len(results)/len(rebalance_turnover) <= 6 else '❌ FAIL'}")

print(f"\n✅ ACTIVE MANAGEMENT:")
print(f"   Turnover when rebalancing: {avg_turnover_when_rebalancing:.1%}")
print(f"   Active management: {'✅ YES' if avg_turnover_when_rebalancing > 0.1 else '❌ NO'}")

# STEP 11: STRATEGY EVALUATION
print(f"\n" + "="*80)
print("STRATEGY EVALUATION & RECOMMENDATIONS")
print("="*80)

print(f"\n🎯 KEY FINDINGS:")
print(f"1. Long-Short Strategy Performance:")
print(f"   • Annualized Return: {ls_metrics['annualized_return']*100:.2f}%")
print(f"   • Annualized Sharpe Ratio: {ls_metrics['annualized_sharpe']:.3f}")
print(f"   • Alpha vs S&P 500: {ls_metrics['alpha_annualized']*100:.2f}% per year")
print(f"   • Maximum Drawdown: {ls_metrics['max_drawdown']*100:.2f}%")

print(f"\n📈 PERFORMANCE ASSESSMENT:")
if ls_metrics['annualized_sharpe'] > 1.0:
    print(f"   ✅ EXCELLENT: Sharpe ratio > 1.0 indicates strong risk-adjusted returns")
elif ls_metrics['annualized_sharpe'] > 0.5:
    print(f"   ✅ GOOD: Sharpe ratio > 0.5 indicates decent risk-adjusted returns")
elif ls_metrics['annualized_sharpe'] > 0:
    print(f"   ⚠️  MODERATE: Positive but low Sharpe ratio")
else:
    print(f"   ❌ POOR: Negative Sharpe ratio indicates poor risk-adjusted performance")

if ls_metrics['alpha_annualized'] > 0.02:
    print(f"   ✅ STRONG ALPHA: Strategy generates significant excess returns vs market")
elif ls_metrics['alpha_annualized'] > 0:
    print(f"   ✅ POSITIVE ALPHA: Strategy generates modest excess returns vs market")
else:
    print(f"   ❌ NEGATIVE ALPHA: Strategy underperforms market on risk-adjusted basis")

print(f"\n🔄 TRADING CONSIDERATIONS:")
print(f"   • Average Turnover: {avg_turnover_when_rebalancing*100:.1f}% per month")
print(f"   • Transaction Costs Impact: {'HIGH' if avg_turnover_when_rebalancing > 0.5 else 'MODERATE' if avg_turnover_when_rebalancing > 0.2 else 'LOW'}")
print(f"   • Implementation Complexity: {'HIGH' if avg_turnover_when_rebalancing > 0.5 else 'MODERATE'}")

# STEP 12: SAVE RESULTS
print(f"\n📁 SAVING RESULTS...")
print("-" * 30)

# Save detailed results
results_summary = {
    'strategy_config': {
        'rebalance_frequency': REBALANCE_FREQUENCY,
        'long_holdings': LONG_HOLDINGS,
        'short_holdings': SHORT_HOLDINGS,
        'total_holdings': LONG_HOLDINGS + SHORT_HOLDINGS,
        'min_holdings_required': MIN_HOLDINGS,
        'max_holdings_required': MAX_HOLDINGS
    },
    'performance_metrics': {
        'strategy': ls_metrics,
        'long_only': long_metrics,
        'short_only': short_metrics,
        'market': market_metrics
    },
    'turnover_analysis': {
        'avg_turnover_when_rebalancing': float(avg_turnover_when_rebalancing),
        'avg_monthly_turnover': float(avg_monthly_turnover),
        'rebalancing_events': len(rebalance_turnover)
    },
    'compliance': {
        'holdings_compliant': MIN_HOLDINGS <= (LONG_HOLDINGS + SHORT_HOLDINGS) <= MAX_HOLDINGS,
        'rebalancing_compliant': len(results)/len(rebalance_turnover) <= 6,
        'active_management': avg_turnover_when_rebalancing > 0.1
    },
    'model_info': best_model_info if 'best_model_info' in locals() else {'model_type': model, 'note': 'fallback_model'},
    'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('portfolio_backtest_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

# Save time series data
results.to_csv('portfolio_backtest_timeseries.csv', index=False)
turnover_df.to_csv('portfolio_turnover_analysis.csv', index=False)

print(f"✅ Results saved:")
print(f"   • portfolio_backtest_results.json (summary metrics)")
print(f"   • portfolio_backtest_timeseries.csv (monthly returns)")
print(f"   • portfolio_turnover_analysis.csv (turnover data)")

print(f"\n" + "="*80)
print("CORRECTED PORTFOLIO BACKTESTING COMPLETE!")
print("="*80)
print(f"Strategy now complies with 50-100 holdings requirement!")
print(f"Total holdings: {LONG_HOLDINGS + SHORT_HOLDINGS} (50 long + 50 short)")
print(f"Rebalancing: {REBALANCE_FREQUENCY}")
print(f"Your {model.upper()} model-based long-short strategy has been fully analyzed.")
print(f"Review the results above to assess the viability of your trading strategy.") 