import pandas as pd
import numpy as np
from datetime import datetime

def show_current_holdings():
    """
    Display current portfolio holdings with ticker symbols and company names
    """
    try:
        # Load detailed holdings
        holdings = pd.read_csv('portfolio_detailed_holdings.csv')
        print(f"✅ Loaded detailed holdings: {len(holdings)} total positions")
        
        # Get the latest rebalancing date
        latest_date = holdings['date'].max()
        latest_holdings = holdings[holdings['date'] == latest_date]
        
        print(f"\n📅 LATEST PORTFOLIO HOLDINGS")
        print(f"Date: {latest_date}")
        print("="*80)
        
        # Separate long and short positions
        long_positions = latest_holdings[latest_holdings['position_type'] == 'LONG'].copy()
        short_positions = latest_holdings[latest_holdings['position_type'] == 'SHORT'].copy()
        
        # Sort by predicted returns
        long_positions = long_positions.sort_values('predicted_return', ascending=False)
        short_positions = short_positions.sort_values('predicted_return', ascending=True)
        
        print(f"\n🟢 LONG POSITIONS ({len(long_positions)} stocks)")
        print("-" * 60)
        print(f"{'Rank':<4} {'Ticker':<8} {'Company Name':<30} {'Pred Ret':<8} {'Actual Ret':<10}")
        print("-" * 60)
        
        for i, (_, row) in enumerate(long_positions.iterrows(), 1):
            ticker = str(row['stock_ticker'])[:7]  # Truncate long tickers
            company = str(row['comp_name'])[:29]   # Truncate long company names
            pred_ret = f"{row['predicted_return']:.3f}"
            actual_ret = f"{row['stock_exret']:.3f}"
            print(f"{i:<4} {ticker:<8} {company:<30} {pred_ret:<8} {actual_ret:<10}")
        
        print(f"\n🔴 SHORT POSITIONS ({len(short_positions)} stocks)")
        print("-" * 60)
        print(f"{'Rank':<4} {'Ticker':<8} {'Company Name':<30} {'Pred Ret':<8} {'Actual Ret':<10}")
        print("-" * 60)
        
        for i, (_, row) in enumerate(short_positions.iterrows(), 1):
            ticker = str(row['stock_ticker'])[:7]  # Truncate long tickers
            company = str(row['comp_name'])[:29]   # Truncate long company names
            pred_ret = f"{row['predicted_return']:.3f}"
            actual_ret = f"{row['stock_exret']:.3f}"
            print(f"{i:<4} {ticker:<8} {company:<30} {pred_ret:<8} {actual_ret:<10}")
        
        # Summary statistics
        print(f"\n📊 PORTFOLIO SUMMARY")
        print("-" * 40)
        print(f"Total Holdings: {len(latest_holdings)}")
        print(f"Long Positions: {len(long_positions)}")
        print(f"Short Positions: {len(short_positions)}")
        print(f"Average Long Predicted Return: {long_positions['predicted_return'].mean():.4f}")
        print(f"Average Short Predicted Return: {short_positions['predicted_return'].mean():.4f}")
        print(f"Average Long Actual Return: {long_positions['stock_exret'].mean():.4f}")
        print(f"Average Short Actual Return: {short_positions['stock_exret'].mean():.4f}")
        
        # Strategy performance for this month
        long_avg_return = long_positions['stock_exret'].mean()
        short_avg_return = short_positions['stock_exret'].mean()
        strategy_return = long_avg_return - short_avg_return
        
        print(f"\n🎯 STRATEGY PERFORMANCE (Latest Month)")
        print("-" * 40)
        print(f"Long Portfolio Return: {long_avg_return:.4f} ({long_avg_return*100:.2f}%)")
        print(f"Short Portfolio Return: {short_avg_return:.4f} ({short_avg_return*100:.2f}%)")
        print(f"Long-Short Return: {strategy_return:.4f} ({strategy_return*100:.2f}%)")
        
        return latest_holdings
        
    except FileNotFoundError:
        print("❌ Error: portfolio_detailed_holdings.csv not found!")
        print("   Please run portfolio_backtest_complete.py first to generate detailed holdings.")
        return None
    except Exception as e:
        print(f"❌ Error loading holdings: {e}")
        return None

def show_top_performers():
    """
    Show top performing stocks across all periods
    """
    try:
        holdings = pd.read_csv('portfolio_detailed_holdings.csv')
        
        print(f"\n🏆 TOP PERFORMING STOCKS (All Time)")
        print("="*80)
        
        # Calculate average performance by stock
        stock_performance = holdings.groupby(['stock_ticker', 'comp_name']).agg({
            'stock_exret': ['mean', 'count'],
            'predicted_return': 'mean',
            'position_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'UNKNOWN'
        }).round(4)
        
        stock_performance.columns = ['avg_actual_return', 'appearances', 'avg_predicted_return', 'most_common_position']
        stock_performance = stock_performance.reset_index()
        
        # Filter stocks that appeared at least 3 times
        frequent_stocks = stock_performance[stock_performance['appearances'] >= 3]
        
        if len(frequent_stocks) > 0:
            # Top performers
            top_performers = frequent_stocks.nlargest(10, 'avg_actual_return')
            
            print(f"\n🥇 TOP 10 BEST PERFORMERS (≥3 appearances)")
            print("-" * 80)
            print(f"{'Ticker':<8} {'Company':<25} {'Avg Return':<10} {'Appearances':<11} {'Position':<8}")
            print("-" * 80)
            
            for _, row in top_performers.iterrows():
                ticker = str(row['stock_ticker'])[:7]
                company = str(row['comp_name'])[:24]
                avg_ret = f"{row['avg_actual_return']:.4f}"
                appearances = f"{row['appearances']}"
                position = str(row['most_common_position'])[:7]
                print(f"{ticker:<8} {company:<25} {avg_ret:<10} {appearances:<11} {position:<8}")
            
            # Worst performers
            worst_performers = frequent_stocks.nsmallest(10, 'avg_actual_return')
            
            print(f"\n🥉 BOTTOM 10 PERFORMERS (≥3 appearances)")
            print("-" * 80)
            print(f"{'Ticker':<8} {'Company':<25} {'Avg Return':<10} {'Appearances':<11} {'Position':<8}")
            print("-" * 80)
            
            for _, row in worst_performers.iterrows():
                ticker = str(row['stock_ticker'])[:7]
                company = str(row['comp_name'])[:24]
                avg_ret = f"{row['avg_actual_return']:.4f}"
                appearances = f"{row['appearances']}"
                position = str(row['most_common_position'])[:7]
                print(f"{ticker:<8} {company:<25} {avg_ret:<10} {appearances:<11} {position:<8}")
        
        else:
            print("No stocks appeared frequently enough for analysis.")
            
    except Exception as e:
        print(f"❌ Error analyzing top performers: {e}")

if __name__ == "__main__":
    print("🏛️ PORTFOLIO HOLDINGS VIEWER")
    print("="*50)
    
    # Show current holdings
    current_holdings = show_current_holdings()
    
    if current_holdings is not None:
        # Show top performers
        show_top_performers()
        
        print(f"\n💡 TIP: The detailed holdings are saved in 'portfolio_detailed_holdings.csv'")
        print("   You can open this file in Excel to see all holdings across all rebalancing dates.") 