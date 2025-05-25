import pandas as pd
import numpy as np
from datetime import datetime

def create_sample_dataset(input_file, output_file, n_stocks=100, min_data_coverage=0.9):
    """
    Create a sample dataset by selecting top N stocks with sufficient data coverage.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to save the sampled dataset
    n_stocks : int
        Number of stocks to select
    min_data_coverage : float
        Minimum required data coverage (0-1)
    """
    print("Reading data in chunks...")
    
    # Read the first chunk to get column names
    chunk_size = 100000
    chunks = pd.read_csv(input_file, chunksize=chunk_size)
    
    # Initialize storage for stock statistics
    stock_stats = {}
    
    # Process chunks to collect statistics
    for chunk in chunks:
        # Assuming there's a 'permno' or similar column for stock identifier
        # and 'date' column for time
        for stock_id in chunk['permno'].unique():
            if stock_id not in stock_stats:
                stock_stats[stock_id] = {
                    'count': 0,
                    'market_cap': chunk[chunk['permno'] == stock_id]['market_equity'].mean()
                }
            stock_stats[stock_id]['count'] += len(chunk[chunk['permno'] == stock_id])
    
    # Calculate total possible observations (assuming monthly data for 20 years)
    total_possible_obs = 12 * 20
    
    # Filter stocks with sufficient coverage
    valid_stocks = {
        stock_id: stats for stock_id, stats in stock_stats.items()
        if stats['count'] / total_possible_obs >= min_data_coverage
    }
    
    # Sort by market cap and select top N stocks
    selected_stocks = sorted(
        valid_stocks.items(),
        key=lambda x: x[1]['market_cap'],
        reverse=True
    )[:n_stocks]
    
    selected_stock_ids = [stock[0] for stock in selected_stocks]
    
    print(f"Selected {len(selected_stock_ids)} stocks with sufficient data coverage")
    
    # Create the sampled dataset
    print("Creating sampled dataset...")
    sampled_data = []
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunk_sample = chunk[chunk['permno'].isin(selected_stock_ids)]
        if not chunk_sample.empty:
            sampled_data.append(chunk_sample)
    
    # Combine all chunks and save
    final_sample = pd.concat(sampled_data, ignore_index=True)
    final_sample.to_csv(output_file, index=False)
    
    print(f"Sampled dataset saved to {output_file}")
    print(f"Final dataset shape: {final_sample.shape}")

if __name__ == "__main__":
    input_file = "C:/Users/abmir/OneDrive/School/McGill/Summer 2/Finance/mma_sample_v2.csv"
    output_file = "sampled_stocks.csv"
    create_sample_dataset(input_file, output_file, n_stocks=100) 