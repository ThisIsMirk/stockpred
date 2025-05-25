import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# AUTOMATICALLY LOAD BEST MODEL
print("LOADING BEST MODEL FOR PORTFOLIO ANALYSIS")
print("="*60)

# Check if best model info exists
if os.path.exists('best_model_info.json'):
    with open('best_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    
    print(f"Best model info found!")
    print(f"   Model: {best_model_info['model_type'].title()}")
    print(f"   Features: {best_model_info['n_features']}")
    print(f"   Overall R²: {best_model_info['overall_r2']:.6f}")
    print(f"   Predictions file: {best_model_info['predictions_file']}")
    
    # Load the best model's predictions
    pred_path = best_model_info['predictions_file']
    model = best_model_info['model_type']
    
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        print(f"Predictions loaded successfully from {pred_path}")
    else:
        print(f"ERROR: Predictions file {pred_path} not found!")
        print("   Please run improved_train_model.py first to generate predictions.")
        exit(1)
        
else:
    print("WARNING: best_model_info.json not found!")
    print("   Falling back to default Ridge model...")
    
    # Fallback to hardcoded Ridge model
    pred_path = "stock_predictions_ridge.csv"  
    model = "ridge"
    
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        print(f"Fallback predictions loaded from {pred_path}")
    else:
        print(f"ERROR: Neither best model info nor fallback file found!")
        print("   Please run improved_train_model.py first to generate predictions.")
        exit(1)

print(f"\nUSING MODEL: {model.upper()}")
print(f"PREDICTIONS FILE: {pred_path}")
print(f"DATA SHAPE: {pred.shape}")
print("="*60)

# Portfolio construction logic (same as main script)
predicted = pred.groupby(["year", "month"])[model]
pred["rank"] = np.floor(
    predicted.transform(lambda s: s.rank()) * 10 / predicted.transform(lambda s: len(s) + 1)
)

# Analyze portfolio composition
print("\nPORTFOLIO STRUCTURE ANALYSIS")
print("="*60)

# 1. Overall statistics
print(f"\nOVERALL STATISTICS:")
print(f"Total observations: {len(pred):,}")
print(f"Unique stocks (permno): {pred['permno'].nunique():,}")
print(f"Time period: {pred['date'].min().strftime('%Y-%m')} to {pred['date'].max().strftime('%Y-%m')}")
print(f"Total months: {pred.groupby(['year', 'month']).ngroups}")

# 2. Portfolio composition by rank
print(f"\nPORTFOLIO COMPOSITION (by decile):")
portfolio_stats = pred.groupby('rank').agg({
    'permno': 'count',
    model: ['mean', 'std', 'min', 'max'],
    'stock_exret': ['mean', 'std']
}).round(4)

portfolio_stats.columns = ['Count', 'Pred_Mean', 'Pred_Std', 'Pred_Min', 'Pred_Max', 
                          'Actual_Mean', 'Actual_Std']
print(portfolio_stats)

# 3. Monthly portfolio sizes
print(f"\nAVERAGE MONTHLY PORTFOLIO SIZES:")
monthly_sizes = pred.groupby(['year', 'month', 'rank']).size().reset_index(name='portfolio_size')
avg_monthly_size = monthly_sizes.groupby('rank')['portfolio_size'].agg(['mean', 'std']).round(1)
print(avg_monthly_size)

# 4. Focus on extreme portfolios (long and short)
print(f"\nEXTREME PORTFOLIOS ANALYSIS:")
print("\nSHORT Portfolio (Rank 0 - Lowest Predicted Returns):")
short_port = pred[pred['rank'] == 0]
print(f"  Average stocks per month: {len(short_port) / pred.groupby(['year', 'month']).ngroups:.0f}")
print(f"  Average predicted return: {short_port[model].mean():.4f}")
print(f"  Average actual return: {short_port['stock_exret'].mean():.4f}")
print(f"  Prediction range: {short_port[model].min():.4f} to {short_port[model].max():.4f}")

print("\nLONG Portfolio (Rank 9 - Highest Predicted Returns):")
long_port = pred[pred['rank'] == 9]
print(f"  Average stocks per month: {len(long_port) / pred.groupby(['year', 'month']).ngroups:.0f}")
print(f"  Average predicted return: {long_port[model].mean():.4f}")
print(f"  Average actual return: {long_port['stock_exret'].mean():.4f}")
print(f"  Prediction range: {long_port[model].min():.4f} to {long_port[model].max():.4f}")

print(f"\nLONG-SHORT SPREAD:")
print(f"  Predicted spread: {long_port[model].mean() - short_port[model].mean():.4f}")
print(f"  Actual spread: {long_port['stock_exret'].mean() - short_port['stock_exret'].mean():.4f}")

# 5. Sample of actual stocks
print(f"\nSAMPLE PORTFOLIO HOLDINGS (Latest Month):")
latest_date = pred['date'].max()
latest_month = pred[pred['date'] == latest_date]

print(f"\nLatest month: {latest_date.strftime('%Y-%m')}")
print(f"Total stocks: {len(latest_month)}")

print("\nTOP 10 LONG POSITIONS (Highest Predicted Returns):")
top_long = latest_month.nlargest(10, model)[['permno', model, 'stock_exret']].round(4)
print(top_long.to_string(index=False))

print("\nTOP 10 SHORT POSITIONS (Lowest Predicted Returns):")
top_short = latest_month.nsmallest(10, model)[['permno', model, 'stock_exret']].round(4)
print(top_short.to_string(index=False))

# 6. Time series of portfolio characteristics
print(f"\nPORTFOLIO EVOLUTION OVER TIME:")
monthly_evolution = pred.groupby(['year', 'month']).agg({
    'permno': 'count',
    model: ['mean', 'std'],
    'stock_exret': ['mean', 'std']
}).round(4)

monthly_evolution.columns = ['Total_Stocks', 'Pred_Mean', 'Pred_Std', 'Actual_Mean', 'Actual_Std']
print("Sample of monthly statistics:")
print(monthly_evolution.head(10))
print("...")
print(monthly_evolution.tail(5))

# 7. Strategy description
print(f"\nTRADING STRATEGY SUMMARY:")
print("="*60)
print("PORTFOLIO TYPE: Long-Short Equity Strategy")
print("UNIVERSE: US Stocks with sufficient data")
print("REBALANCING: Monthly")
print(f"SIGNAL: Machine Learning Predicted Returns ({model.title()} Regression)")
if 'best_model_info' in locals():
    print(f"MODEL PERFORMANCE: Overall R² = {best_model_info['overall_r2']:.6f}")
    print(f"FEATURES USED: {best_model_info['n_features']} selected features")
print("\nSTRATEGY MECHANICS:")
print("1. Each month, rank ALL stocks by predicted returns")
print("2. Divide into 10 equal-sized portfolios (deciles)")
print("3. Portfolio 1 (rank 0): WORST predicted returns → SHORT")
print("4. Portfolio 10 (rank 9): BEST predicted returns → LONG")
print("5. Portfolio 11: LONG - SHORT (the strategy return)")
print("\nPORTFOLIO WEIGHTS:")
print("- Equal-weighted within each portfolio")
print("- No leverage (dollar-neutral long-short)")
print("- Monthly rebalancing based on new predictions")

print(f"\nINTERPRETATION:")
print("Your model is trying to identify:")
print("• Stocks that will OUTPERFORM → BUY (Long portfolio)")
print("• Stocks that will UNDERPERFORM → SELL SHORT (Short portfolio)")
print("• Profit comes from the SPREAD between long and short returns")

# 8. Model selection summary
if 'best_model_info' in locals():
    print(f"\nSELECTED MODEL DETAILS:")
    print("="*60)
    print(f"Model Type: {best_model_info['model_type'].title()}")
    print(f"Number of Features: {best_model_info['n_features']}")
    print(f"Overall Out-of-Sample R²: {best_model_info['overall_r2']:.6f}")
    print(f"Mean Period R²: {best_model_info['mean_period_r2']:.6f}")
    print(f"Positive R² Periods: {best_model_info['positive_r2_periods']}/{best_model_info['total_periods']}")
    print(f"Target Achieved (R² ≥ 0.01): {best_model_info['target_achieved_periods']}/{best_model_info['total_periods']} periods")
    
    if best_model_info['overall_r2'] > 0:
        print(f"\nMODEL QUALITY: Good - Positive out-of-sample R²")
    else:
        print(f"\nMODEL QUALITY: Poor - Negative out-of-sample R²")
        print("   Consider improving the model before live trading")
else:
    print(f"\nUsing fallback model - run improved_train_model.py for optimal selection")   