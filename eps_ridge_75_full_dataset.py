import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from tqdm import tqdm
import time
import os
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

def prepare_data_enhanced(data, n_features=75):
    """
    Enhanced data preparation with outlier handling and feature selection.
    Optimized for EPS prediction with Ridge regression and 75 features.
    """
    # Using 'eps_actual' as the target variable
    y = data['eps_actual']
    
    # Drop non-feature columns
    columns_to_drop = ['eps_actual', 'permno', 'date']
    
    # Drop forward-looking columns that would cause data leakage
    # Note: We exclude 'stock_exret' from features but keep it in data for portfolio backtesting
    forward_looking_columns = ['eps_medest', 'eps_meanest', 'eps_stdevest', 'stock_exret']
    # Only drop columns that actually exist in the data
    existing_forward_looking = [col for col in forward_looking_columns if col in data.columns]
    columns_to_drop.extend(existing_forward_looking)
    
    # Also drop any string/object columns that can't be used as features
    string_columns = data.select_dtypes(include=['object']).columns.tolist()
    columns_to_drop.extend(string_columns)
    
    # Create feature matrix
    X = data.drop(columns_to_drop, axis=1)
    
    # Handle any remaining non-numeric values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values with median
    X = X.fillna(X.median())
    
    # Remove features with zero variance
    variance_threshold = X.var()
    zero_var_features = variance_threshold[variance_threshold == 0].index
    X = X.drop(zero_var_features, axis=1)
    
    # Handle outliers using robust scaling
    # Cap extreme outliers at 5th and 95th percentiles
    for col in X.columns:
        q05, q95 = X[col].quantile([0.05, 0.95])
        X[col] = X[col].clip(lower=q05, upper=q95)
    
    # Feature selection using univariate selection
    if len(X.columns) > n_features:
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    print(f"Features used: {X.shape[1]} columns (after feature selection)")
    print(f"Forward-looking columns dropped: {existing_forward_looking}")
    print(f"String columns dropped: {string_columns}")
    if len(zero_var_features) > 0:
        print(f"Zero variance features dropped: {len(zero_var_features)}")
    
    return X, y

def optimize_ridge_hyperparameters(X_train, y_train, cv_folds=3):
    """
    Optimize Ridge hyperparameters using cross-validation.
    """
    # Test a range of alpha values for Ridge
    alphas = np.logspace(-4, 2, 20)  # From 0.0001 to 100
    model = RidgeCV(alphas=alphas, cv=cv_folds)
    model.fit(X_train, y_train)
    return model, model.alpha_

def train_eps_ridge_model(data):
    """
    Train Ridge model for EPS prediction with 75 features using expanding window.
    """
    # Handle different date formats
    if data['date'].dtype == 'int64':
        data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    else:
        data['date'] = pd.to_datetime(data['date'])
    
    print("Date range in data:", data['date'].min(), "to", data['date'].max())
    
    # Initialize results storage
    results = []
    all_predictions = []  # Store individual stock predictions
    
    # EXPANDING WINDOW SCHEDULE:
    # Initial periods
    train_start = pd.to_datetime('2000-01-01')  # Training start NEVER changes
    train_end = pd.to_datetime('2007-12-31')    # Training end EXPANDS each year
    val_start = pd.to_datetime('2008-01-01')    # Validation rolls forward each year
    val_end = pd.to_datetime('2009-12-31')      # Validation rolls forward each year
    test_start = pd.to_datetime('2010-01-01')   # Test rolls forward each year
    test_end = pd.to_datetime('2010-12-31')     # Test rolls forward each year
    
    # Calculate total number of iterations
    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2023-12-31')
    total_years = (end_date - start_date).days / 365.25
    total_iterations = int(total_years)
    
    # Create progress bar
    pbar = tqdm(total=total_iterations, desc="Training EPS Ridge Model (75 features)")
    
    # Start timing
    start_time = time.time()
    
    while test_end <= pd.to_datetime('2023-12-31'):
        iteration_start = time.time()
        
        # Prepare data for each period
        train_data = data[(data['date'] >= train_start) & (data['date'] <= train_end)]
        val_data = data[(data['date'] >= val_start) & (data['date'] <= val_end)]
        test_data = data[(data['date'] >= test_start) & (data['date'] <= test_end)]
        
        if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
            print("WARNING: Empty data in one or more periods!")
            # Use expanding window logic even for error cases
            train_end = train_end + pd.DateOffset(years=1)
            val_start = val_start + pd.DateOffset(years=1)
            val_end = val_end + pd.DateOffset(years=1)
            test_start = test_start + pd.DateOffset(years=1)
            test_end = test_start + pd.DateOffset(years=1) - pd.DateOffset(days=1)
            pbar.update(1)
            continue
        
        # Prepare features and target with enhanced preprocessing
        X_train, y_train = prepare_data_enhanced(train_data, n_features=75)
        X_val, y_val = prepare_data_enhanced(val_data, n_features=75)
        X_test, y_test = prepare_data_enhanced(test_data, n_features=75)
        
        # Ensure all datasets have the same features
        common_features = X_train.columns.intersection(X_val.columns).intersection(X_test.columns)
        X_train = X_train[common_features]
        X_val = X_val[common_features]
        X_test = X_test[common_features]
        
        if len(common_features) == 0:
            print("WARNING: No common features found!")
            # Use expanding window logic even for error cases
            train_end = train_end + pd.DateOffset(years=1)
            val_start = val_start + pd.DateOffset(years=1)
            val_end = val_end + pd.DateOffset(years=1)
            test_start = test_start + pd.DateOffset(years=1)
            test_end = test_start + pd.DateOffset(years=1) - pd.DateOffset(days=1)
            pbar.update(1)
            continue
        
        # Scale features using RobustScaler (better for outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Optimize hyperparameters
        model, best_alpha = optimize_ridge_hyperparameters(X_train_scaled, y_train)
        
        # Make predictions
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Store individual predictions for portfolio analysis
        prediction_data = {
            'permno': test_data['permno'].values,
            'date': test_data['date'].values,
            'year': test_data['date'].dt.year.values,
            'month': test_data['date'].dt.month.values,
            'eps_actual': y_test.values,  # actual EPS
            'ridge': test_pred  # predicted EPS
        }
        
        # Add stock returns for portfolio backtesting
        if 'stock_exret' in test_data.columns:
            prediction_data['stock_exret'] = test_data['stock_exret'].values
        
        # Add ticker and company info if available
        if 'stock_ticker' in test_data.columns:
            prediction_data['stock_ticker'] = test_data['stock_ticker'].values
        if 'comp_name' in test_data.columns:
            prediction_data['comp_name'] = test_data['comp_name'].values
            
        test_predictions = pd.DataFrame(prediction_data)
        all_predictions.append(test_predictions)
        
        # Calculate metrics
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        # Store results
        results.append({
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'val_start': val_start.strftime('%Y-%m-%d'),
            'val_end': val_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'val_r2': val_r2,
            'test_r2': test_r2,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'best_alpha': best_alpha,
            'n_features': len(common_features),
            'train_size': len(train_data),
            'test_size': len(test_data)
        })
        
        # Update dates for next iteration (EXPANDING WINDOW)
        train_end = train_end + pd.DateOffset(years=1)  # Expand training by 1 year
        val_start = val_start + pd.DateOffset(years=1)  # Roll validation forward by 1 year
        val_end = val_end + pd.DateOffset(years=1)      # Roll validation forward by 1 year
        test_start = test_start + pd.DateOffset(years=1)  # Roll test forward by 1 year
        test_end = test_start + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        
        # Update progress bar
        pbar.update(1)
        
        # Calculate and display iteration time
        iteration_time = time.time() - iteration_start
        pbar.set_postfix({
            'iteration_time': f'{iteration_time:.2f}s',
            'val_r2': f'{val_r2:.4f}',
            'test_r2': f'{test_r2:.4f}',
            'alpha': f'{best_alpha:.4f}',
            'features': len(common_features)
        })
    
    # Close progress bar
    pbar.close()
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f"\nTotal training time: {timedelta(seconds=int(total_time))}")
    
    # Save individual predictions for portfolio analysis
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Create eps_predictions directory if it doesn't exist
        os.makedirs("eps_predictions", exist_ok=True)
        
        predictions_filename = "eps_predictions/eps_predictions_ridge.csv"
        all_predictions_df.to_csv(predictions_filename, index=False)
        print(f"Individual EPS predictions saved to: {predictions_filename}")
        print(f"Predictions shape: {all_predictions_df.shape}")
        
        # Check what columns were included
        has_ticker = 'stock_ticker' in all_predictions_df.columns
        has_company = 'comp_name' in all_predictions_df.columns
        print(f"Columns included: {list(all_predictions_df.columns)}")
        if has_ticker and has_company:
            print("✅ Ticker symbols and company names included!")
        elif has_ticker:
            print("⚠️  Ticker symbols included, but company names missing")
        elif has_company:
            print("⚠️  Company names included, but ticker symbols missing")
        else:
            print("⚠️  No ticker or company information available")
        
        # Calculate overall out-of-sample R²
        overall_r2 = r2_score(all_predictions_df['eps_actual'], all_predictions_df['ridge'])
        overall_mse = mean_squared_error(all_predictions_df['eps_actual'], all_predictions_df['ridge'])
        
        print(f"\n🎯 OVERALL OUT-OF-SAMPLE PERFORMANCE (EPS RIDGE 75):")
        print(f"  Total test observations: {len(all_predictions_df):,}")
        print(f"  Overall R²: {overall_r2:.6f}")
        print(f"  Overall MSE: {overall_mse:.6f}")
        print(f"  Overall RMSE: {np.sqrt(overall_mse):.6f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Return both results and overall R² if predictions exist
    if all_predictions:
        overall_r2 = r2_score(all_predictions_df['eps_actual'], all_predictions_df['ridge'])
        return results_df, overall_r2, all_predictions_df
    else:
        return results_df, None, None

def save_model_results(results_df, overall_r2, predictions_df):
    """
    Save model results and create best model info for portfolio backtesting.
    """
    # Create eps_predictions directory if it doesn't exist
    os.makedirs("eps_predictions", exist_ok=True)
    
    # Save detailed results
    results_df.to_csv("eps_predictions/eps_ridge_75_detailed_results.csv", index=False)
    
    # Create model performance summary
    model_summary = {
        'model': 'Ridge_EPS',
        'n_features': 75,
        'overall_r2': overall_r2,
        'overall_mse': mean_squared_error(predictions_df['eps_actual'], predictions_df['ridge']),
        'mean_period_r2': results_df['test_r2'].mean(),
        'median_period_r2': results_df['test_r2'].median(),
        'std_period_r2': results_df['test_r2'].std(),
        'min_period_r2': results_df['test_r2'].min(),
        'max_period_r2': results_df['test_r2'].max(),
        'positive_r2_periods': (results_df['test_r2'] > 0).sum(),
        'total_periods': len(results_df),
        'target_achieved_periods': (results_df['test_r2'] >= 0.01).sum(),
        'mean_alpha': results_df['best_alpha'].mean(),
        'total_test_observations': len(predictions_df)
    }
    
    # Save summary
    summary_df = pd.DataFrame([model_summary])
    summary_df.to_csv("eps_predictions/eps_model_performance_summary.csv", index=False)
    
    # Create best model info for portfolio backtesting (compatible with existing code)
    best_model_info = {
        'model_type': 'ridge',
        'n_features': 75,
        'overall_r2': overall_r2,
        'overall_mse': model_summary['overall_mse'],
        'mean_period_r2': model_summary['mean_period_r2'],
        'positive_r2_periods': int(model_summary['positive_r2_periods']),
        'target_achieved_periods': int(model_summary['target_achieved_periods']),
        'total_periods': int(model_summary['total_periods']),
        'predictions_file': "eps_predictions/eps_predictions_ridge.csv"
    }
    
    # Save best model info
    with open('eps_predictions/best_eps_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   • eps_predictions/eps_predictions_ridge.csv - Individual predictions")
    print(f"   • eps_predictions/eps_ridge_75_detailed_results.csv - Period-by-period results")
    print(f"   • eps_predictions/eps_model_performance_summary.csv - Performance summary")
    print(f"   • eps_predictions/best_eps_model_info.json - Best model info for portfolio backtesting")
    
    return best_model_info

if __name__ == "__main__":
    print(f"🚀 Starting EPS Ridge Model (75 features) on Full Dataset")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load the full dataset
    print("📂 Loading full dataset...")
    data_path = "C:/Users/abmir/OneDrive/School/McGill/Summer 2/Finance/mma_sample_v2.csv"
    
    try:
        data = pd.read_csv(data_path)
        print(f"✅ Data loaded successfully. Shape: {data.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find file at {data_path}")
        print("Please check the file path and try again.")
        exit(1)
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        exit(1)
    
    # Check if eps_actual column exists and has data
    if 'eps_actual' not in data.columns:
        print("❌ ERROR: 'eps_actual' column not found in the dataset!")
        print(f"Available columns: {list(data.columns)}")
        exit(1)
    
    # Check for missing values in eps_actual
    eps_missing = data['eps_actual'].isna().sum()
    eps_total = len(data)
    eps_coverage = (eps_total - eps_missing) / eps_total * 100
    
    print(f"\n📊 EPS Data Quality Check:")
    print(f"  Total observations: {eps_total:,}")
    print(f"  Missing EPS values: {eps_missing:,}")
    print(f"  EPS coverage: {eps_coverage:.1f}%")
    print(f"  EPS range: {data['eps_actual'].min():.4f} to {data['eps_actual'].max():.4f}")
    print(f"  EPS mean: {data['eps_actual'].mean():.4f}")
    print(f"  EPS std: {data['eps_actual'].std():.4f}")
    
    if eps_coverage < 50:
        print("⚠️  WARNING: Low EPS coverage! Results may be unreliable.")
    
    # Filter out rows with missing EPS data
    data_clean = data.dropna(subset=['eps_actual'])
    print(f"After removing missing EPS: {len(data_clean):,} observations ({len(data_clean)/len(data)*100:.1f}% retained)")
    
    # Train the model
    print(f"\n🤖 Training EPS Ridge Model with 75 features...")
    print("="*60)
    
    results_df, overall_r2, predictions_df = train_eps_ridge_model(data_clean)
    
    if overall_r2 is not None:
        print(f"\n🎯 MODEL PERFORMANCE SUMMARY:")
        print("="*50)
        print(f"📊 Overall Out-of-Sample R²: {overall_r2:.6f}")
        print(f"📈 Mean Period R²: {results_df['test_r2'].mean():.6f}")
        print(f"📉 Min Period R²: {results_df['test_r2'].min():.6f}")
        print(f"📈 Max Period R²: {results_df['test_r2'].max():.6f}")
        print(f"✅ Positive R² Periods: {(results_df['test_r2'] > 0).sum()}/{len(results_df)}")
        print(f"🎯 Target Achieved (R² ≥ 0.01): {(results_df['test_r2'] >= 0.01).sum()}/{len(results_df)}")
        print(f"📊 Total Test Observations: {len(predictions_df):,}")
        
        # Save results
        best_model_info = save_model_results(results_df, overall_r2, predictions_df)
        
        print(f"\n🎉 SUCCESS! EPS Ridge model training completed successfully!")
        print(f"✅ The model is ready for portfolio backtesting with portfolio_backtest_eps.py")
        print(f"✅ Best model info saved for automatic selection in portfolio backtesting")
        
        if overall_r2 > 0.15:
            print(f"🏆 EXCELLENT PERFORMANCE! R² > 0.15 indicates strong predictive power")
        elif overall_r2 > 0.10:
            print(f"👍 GOOD PERFORMANCE! R² > 0.10 indicates decent predictive power")
        elif overall_r2 > 0.05:
            print(f"⚠️  MODERATE PERFORMANCE: R² > 0.05 but could be improved")
        else:
            print(f"⚠️  LOW PERFORMANCE: Consider feature engineering or different approaches")
            
    else:
        print("❌ ERROR: Model training failed - no predictions generated")
        exit(1)
    
    print(f"\n🏁 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("🚀 Ready to run portfolio_backtest_eps.py for strategy evaluation!") 