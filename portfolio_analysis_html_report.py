import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def create_html_report():
    # Load predictions (same as original)
    pred_path = "stock_predictions_ridge.csv"  
    pred = pd.read_csv(pred_path, parse_dates=["date"])
    model = "ridge"
    
    # Portfolio construction logic (same as original)
    predicted = pred.groupby(["year", "month"])[model]
    pred["rank"] = np.floor(
        predicted.transform(lambda s: s.rank()) * 10 / predicted.transform(lambda s: len(s) + 1)
    )
    
    # Start building HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 40px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 40px;
        }}
        h2 {{
            color: #34495e;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-top: 40px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f4fd;
        }}
        .highlight {{
            background-color: #e8f4fd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            margin-top: 40px;
        }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 20px 0;
        }}
        .metric {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .metric strong {{
            color: #2c3e50;
        }}
        ol, ul {{
            padding-left: 20px;
        }}
        li {{
            margin: 8px 0;
        }}
        .performance-highlight {{
            background-color: #d5f4e6;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 15px 0;
        }}
        .warning-highlight {{
            background-color: #fdeaea;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ Portfolio Analysis Report</h1>
        <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
"""
    
    # 1. Overall Statistics
    total_months = pred.groupby(['year', 'month']).ngroups
    html_content += f"""
        <h2>📊 Overall Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div>Total Observations</div>
                <div class="stat-value">{len(pred):,}</div>
            </div>
            <div class="stat-card">
                <div>Unique Stocks (permno)</div>
                <div class="stat-value">{pred['permno'].nunique():,}</div>
            </div>
            <div class="stat-card">
                <div>Time Period</div>
                <div class="stat-value">{pred['date'].min().strftime('%Y-%m')} to {pred['date'].max().strftime('%Y-%m')}</div>
            </div>
            <div class="stat-card">
                <div>Total Months</div>
                <div class="stat-value">{total_months}</div>
            </div>
        </div>
"""
    
    # 2. Portfolio Composition
    portfolio_stats = pred.groupby('rank').agg({
        'permno': 'count',
        model: ['mean', 'std', 'min', 'max'],
        'stock_exret': ['mean', 'std']
    }).round(4)
    
    portfolio_stats.columns = ['Count', 'Pred_Mean', 'Pred_Std', 'Pred_Min', 'Pred_Max', 
                              'Actual_Mean', 'Actual_Std']
    
    # Add rank labels for better readability
    portfolio_stats.index = [f"Rank {i} {'(SHORT)' if i == 0 else '(LONG)' if i == 9 else ''}" for i in portfolio_stats.index]
    
    html_content += f"""
        <h2>📈 Portfolio Composition by Decile</h2>
        <p>Each rank represents a decile portfolio based on predicted returns. Rank 0 = worst predictions (SHORT), Rank 9 = best predictions (LONG).</p>
        {portfolio_stats.to_html(classes='table', table_id='portfolio-table')}
"""
    
    # 3. Monthly Portfolio Sizes
    monthly_sizes = pred.groupby(['year', 'month', 'rank']).size().reset_index(name='portfolio_size')
    avg_monthly_size = monthly_sizes.groupby('rank')['portfolio_size'].agg(['mean', 'std']).round(1)
    avg_monthly_size.index = [f"Rank {i}" for i in avg_monthly_size.index]
    
    html_content += f"""
        <h2>📅 Average Monthly Portfolio Sizes</h2>
        <p>Average number of stocks in each portfolio per month:</p>
        {avg_monthly_size.to_html(classes='table')}
"""
    
    # 4. Extreme Portfolios Analysis
    short_port = pred[pred['rank'] == 0]
    long_port = pred[pred['rank'] == 9]
    
    pred_spread = long_port[model].mean() - short_port[model].mean()
    actual_spread = long_port['stock_exret'].mean() - short_port['stock_exret'].mean()
    
    html_content += f"""
        <h2>🎯 Extreme Portfolios Analysis</h2>
        
        <div class="two-column">
            <div class="stat-card">
                <h3>SHORT Portfolio (Rank 0)</h3>
                <div class="metric"><strong>Avg stocks/month:</strong> {len(short_port) / total_months:.0f}</div>
                <div class="metric"><strong>Avg predicted return:</strong> <span class="negative">{short_port[model].mean():.4f}</span></div>
                <div class="metric"><strong>Avg actual return:</strong> <span class="negative">{short_port['stock_exret'].mean():.4f}</span></div>
                <div class="metric"><strong>Prediction range:</strong> {short_port[model].min():.4f} to {short_port[model].max():.4f}</div>
            </div>
            
            <div class="stat-card">
                <h3>LONG Portfolio (Rank 9)</h3>
                <div class="metric"><strong>Avg stocks/month:</strong> {len(long_port) / total_months:.0f}</div>
                <div class="metric"><strong>Avg predicted return:</strong> <span class="positive">{long_port[model].mean():.4f}</span></div>
                <div class="metric"><strong>Avg actual return:</strong> <span class="positive">{long_port['stock_exret'].mean():.4f}</span></div>
                <div class="metric"><strong>Prediction range:</strong> {long_port[model].min():.4f} to {long_port[model].max():.4f}</div>
            </div>
        </div>
        
        <div class="{'performance-highlight' if actual_spread > 0 else 'warning-highlight'}">
            <h3>📊 Long-Short Spread Analysis:</h3>
            <div class="metric"><strong>Predicted spread:</strong> {pred_spread:.4f}</div>
            <div class="metric"><strong>Actual spread:</strong> <span class="{'positive' if actual_spread > 0 else 'negative'}">{actual_spread:.4f}</span></div>
            <div class="metric"><strong>Strategy Performance:</strong> {'✅ Profitable' if actual_spread > 0 else '❌ Unprofitable'} 
                {'(Model predictions align with reality)' if (pred_spread > 0 and actual_spread > 0) or (pred_spread < 0 and actual_spread < 0) else '(Model predictions misaligned)'}</div>
        </div>
"""
    
    # 5. Latest Month Holdings
    latest_date = pred['date'].max()
    latest_month = pred[pred['date'] == latest_date]
    
    top_long = latest_month.nlargest(10, model)[['permno', model, 'stock_exret']].round(4)
    top_short = latest_month.nsmallest(10, model)[['permno', model, 'stock_exret']].round(4)
    
    # Rename columns for better display
    top_long.columns = ['Stock ID (permno)', 'Predicted Return', 'Actual Return']
    top_short.columns = ['Stock ID (permno)', 'Predicted Return', 'Actual Return']
    
    html_content += f"""
        <h2>📋 Latest Month Portfolio Holdings</h2>
        <p><strong>Latest month:</strong> {latest_date.strftime('%Y-%m')} | <strong>Total stocks:</strong> {len(latest_month):,}</p>
        
        <div class="two-column">
            <div>
                <h3>🟢 Top 10 Long Positions</h3>
                <p><em>Highest predicted returns</em></p>
                {top_long.to_html(classes='table', index=False)}
            </div>
            <div>
                <h3>🔴 Top 10 Short Positions</h3>
                <p><em>Lowest predicted returns</em></p>
                {top_short.to_html(classes='table', index=False)}
            </div>
        </div>
"""
    
    # 6. Portfolio Evolution Over Time
    monthly_evolution = pred.groupby(['year', 'month']).agg({
        'permno': 'count',
        model: ['mean', 'std'],
        'stock_exret': ['mean', 'std']
    }).round(4)
    
    monthly_evolution.columns = ['Total_Stocks', 'Pred_Mean', 'Pred_Std', 'Actual_Mean', 'Actual_Std']
    
    # Create a date index for better display
    monthly_evolution.index = pd.to_datetime(monthly_evolution.index.map(lambda x: f"{x[0]}-{x[1]:02d}"))
    
    html_content += f"""
        <h2>📈 Portfolio Evolution Over Time</h2>
        <p>Monthly aggregated statistics across all portfolios:</p>
        
        <h3>Recent Months (Last 10):</h3>
        {monthly_evolution.tail(10).to_html(classes='table')}
        
        <h3>Early Months (First 10):</h3>
        {monthly_evolution.head(10).to_html(classes='table')}
"""
    
    # 7. Strategy Summary
    html_content += f"""
        <h2>🎯 Trading Strategy Summary</h2>
        <div class="summary-box">
            <div class="stats-grid">
                <div class="metric"><strong>Portfolio Type:</strong> Long-Short Equity Strategy</div>
                <div class="metric"><strong>Universe:</strong> US Stocks with sufficient data</div>
                <div class="metric"><strong>Rebalancing:</strong> Monthly</div>
                <div class="metric"><strong>Signal:</strong> Machine Learning Predicted Returns (Ridge Regression)</div>
            </div>
            
            <h3>🔧 Strategy Mechanics:</h3>
            <ol>
                <li>Each month, rank ALL stocks by predicted returns</li>
                <li>Divide into 10 equal-sized portfolios (deciles)</li>
                <li>Portfolio 1 (rank 0): WORST predicted returns → <strong>SHORT</strong></li>
                <li>Portfolio 10 (rank 9): BEST predicted returns → <strong>LONG</strong></li>
                <li>Portfolio 11: LONG - SHORT (the strategy return)</li>
            </ol>
            
            <h3>⚖️ Portfolio Weights:</h3>
            <ul>
                <li>Equal-weighted within each portfolio</li>
                <li>No leverage (dollar-neutral long-short)</li>
                <li>Monthly rebalancing based on new predictions</li>
            </ul>
            
            <h3>💡 Interpretation:</h3>
            <div class="highlight">
                <p><strong>Your model is trying to identify:</strong></p>
                <ul>
                    <li>🟢 Stocks that will <strong>OUTPERFORM</strong> → BUY (Long portfolio)</li>
                    <li>🔴 Stocks that will <strong>UNDERPERFORM</strong> → SELL SHORT (Short portfolio)</li>
                    <li>💰 Profit comes from the <strong>SPREAD</strong> between long and short returns</li>
                </ul>
            </div>
        </div>
        
        <div class="timestamp">
            <p>📄 Report generated from: <code>{pred_path}</code></p>
            <p>🤖 Model: {model.upper()} Regression</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save the report
    output_file = f"portfolio_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML Report successfully created!")
    print(f"📁 File saved as: {output_file}")
    print(f"🌐 Open the file in your browser to view the beautiful report!")
    print(f"📊 The report includes all portfolio analysis with professional styling.")
    
    return output_file

if __name__ == "__main__":
    create_html_report() 