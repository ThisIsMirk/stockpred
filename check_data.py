import pandas as pd
import numpy as np

def check_data():
    """
    Diagnostic function to check the data structure and dates
    """
    print("Loading data...")
    data = pd.read_csv("sampled_stocks.csv")
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Check if we have the expected columns
    print(f"\nChecking for key columns:")
    print(f"- 'date' column exists: {'date' in data.columns}")
    print(f"- 'year' column exists: {'year' in data.columns}")
    print(f"- 'month' column exists: {'month' in data.columns}")
    print(f"- 'stock_exret' column exists: {'stock_exret' in data.columns}")
    
    # Look at the first few rows
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    # Check date column specifically
    if 'date' in data.columns:
        print(f"\nDate column info:")
        print(f"- Data type: {data['date'].dtype}")
        print(f"- First 10 values: {data['date'].head(10).tolist()}")
        print(f"- Unique values count: {data['date'].nunique()}")
        print(f"- Min value: {data['date'].min()}")
        print(f"- Max value: {data['date'].max()}")
        
        # Try to convert to datetime
        try:
            dates_converted = pd.to_datetime(data['date'])
            print(f"- Conversion successful!")
            print(f"- Date range after conversion: {dates_converted.min()} to {dates_converted.max()}")
        except Exception as e:
            print(f"- Conversion failed: {e}")
    
    # Check year/month columns if they exist
    if 'year' in data.columns and 'month' in data.columns:
        print(f"\nYear/Month column info:")
        print(f"- Year range: {data['year'].min()} to {data['year'].max()}")
        print(f"- Month range: {data['month'].min()} to {data['month'].max()}")
        print(f"- Sample year/month combinations:")
        print(data[['year', 'month']].head(10))

if __name__ == "__main__":
    check_data() 