import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
from tqdm import tqdm
import time
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

def prepare_data_enhanced(data, n_features=50):
    """
    Enhanced data preparation with outlier handling and feature selection.
    """
    # Assuming 'stock_exret' is the target variable
    y = data['stock_exret']
    
    # Drop non-feature columns
    columns_to_drop = ['stock_exret', 'permno', 'date']
    
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
    print(f"String columns dropped: {string_columns}")
    if len(zero_var_features) > 0:
        print(f"Zero variance features dropped: {len(zero_var_features)}")
    
    return X, y

def optimize_hyperparameters(X_train, y_train, model_type='lasso', cv_folds=3):
    """
    Optimize hyperparameters using cross-validation.
    """
    if model_type == 'lasso':
        # Test a range of alpha values for Lasso
        alphas = np.logspace(-4, 1, 20)  # From 0.0001 to 10
        model = LassoCV(alphas=alphas, cv=cv_folds, random_state=42, max_iter=2000)
    else:
        # Test a range of alpha values for Ridge
        alphas = np.logspace(-4, 2, 20)  # From 0.0001 to 100
        model = RidgeCV(alphas=alphas, cv=cv_folds)
    
    model.fit(X_train, y_train)
    return model, model.alpha_

def train_and_evaluate_enhanced(data, model_type='lasso', n_features=50):
    """
    Enhanced training with hyperparameter tuning and better preprocessing.
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
    # Iteration 1: Train 2000-2007, Val 2008-2009, Test 2010
    # Iteration 2: Train 2000-2008, Val 2009-2010, Test 2011  
    # Iteration 3: Train 2000-2009, Val 2010-2011, Test 2012
    # ... and so on until Test 2023
    
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
    pbar = tqdm(total=total_iterations, desc=f"Training enhanced {model_type} model")
    
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
        X_train, y_train = prepare_data_enhanced(train_data, n_features=n_features)
        X_val, y_val = prepare_data_enhanced(val_data, n_features=n_features)
        X_test, y_test = prepare_data_enhanced(test_data, n_features=n_features)
        
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
        model, best_alpha = optimize_hyperparameters(X_train_scaled, y_train, model_type)
        
        # Make predictions
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Store individual predictions for portfolio analysis
        prediction_data = {
            'permno': test_data['permno'].values,
            'date': test_data['date'].values,
            'year': test_data['date'].dt.year.values,
            'month': test_data['date'].dt.month.values,
            'stock_exret': y_test.values,  # actual returns
            model_type: test_pred  # predicted returns for this model
        }
        
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
        # Training window expands by one year (keeps start date, extends end date)
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
        
        # Create stock_ret_predictions directory if it doesn't exist
        os.makedirs("stock_ret_predictions", exist_ok=True)
        
        predictions_filename = f"stock_ret_predictions/stock_predictions_{model_type}.csv"
        all_predictions_df.to_csv(predictions_filename, index=False)
        print(f"Individual predictions saved to: {predictions_filename}")
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
        overall_r2 = r2_score(all_predictions_df['stock_exret'], all_predictions_df[model_type])
        overall_mse = mean_squared_error(all_predictions_df['stock_exret'], all_predictions_df[model_type])
        
        print(f"\n🎯 OVERALL OUT-OF-SAMPLE PERFORMANCE:")
        print(f"  Total test observations: {len(all_predictions_df):,}")
        print(f"  Overall R²: {overall_r2:.6f}")
        print(f"  Overall MSE: {overall_mse:.6f}")
        print(f"  Overall RMSE: {np.sqrt(overall_mse):.6f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Return both results and overall R² if predictions exist
    if all_predictions:
        overall_r2 = r2_score(all_predictions_df['stock_exret'], all_predictions_df[model_type])
        return results_df, overall_r2
    else:
        return results_df, None

def analyze_results(results_df, model_name, overall_r2=None):
    """
    Analyze the results and provide insights.
    """
    print(f"\n{model_name} Results Analysis:")
    print("="*50)
    
    # Overall performance (if provided)
    if overall_r2 is not None:
        print(f"📊 OVERALL OUT-OF-SAMPLE R²: {overall_r2:.6f}")
        print("-" * 50)
    
    # Basic statistics
    test_r2_mean = results_df['test_r2'].mean()
    test_r2_std = results_df['test_r2'].std()
    test_r2_median = results_df['test_r2'].median()
    positive_r2_count = (results_df['test_r2'] > 0).sum()
    target_r2_count = (results_df['test_r2'] >= 0.01).sum()
    
    print(f"Period-by-Period R² Statistics:")
    print(f"  Mean: {test_r2_mean:.4f}")
    print(f"  Median: {test_r2_median:.4f}")
    print(f"  Std: {test_r2_std:.4f}")
    print(f"  Min: {results_df['test_r2'].min():.4f}")
    print(f"  Max: {results_df['test_r2'].max():.4f}")
    print(f"  Positive R²: {positive_r2_count}/{len(results_df)} periods")
    print(f"  R² ≥ 0.01: {target_r2_count}/{len(results_df)} periods")
    
    # Alpha statistics
    print(f"\nHyperparameter Statistics:")
    print(f"  Alpha mean: {results_df['best_alpha'].mean():.4f}")
    print(f"  Alpha range: {results_df['best_alpha'].min():.4f} - {results_df['best_alpha'].max():.4f}")
    
    return results_df

if __name__ == "__main__":
    print(f"Starting enhanced training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load the sampled data
    print("Loading data...")
    data = pd.read_csv("sampled_stocks.csv")
    print(f"Data loaded. Shape: {data.shape}")
    
    # Try different feature selection sizes
    feature_sizes = [30, 50, 75]
    
    # Initialize list to store overall model performance
    model_summary = []
    
    for n_features in feature_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_features} features")
        print(f"{'='*60}")
        
        # Train and evaluate enhanced Lasso model
        print(f"\nTraining Enhanced Lasso model with {n_features} features...")
        lasso_results, lasso_overall_r2 = train_and_evaluate_enhanced(data, model_type='lasso', n_features=n_features)
        lasso_analysis = analyze_results(lasso_results, f"Enhanced Lasso ({n_features} features)", lasso_overall_r2)
        
        # Train and evaluate enhanced Ridge model
        print(f"\nTraining Enhanced Ridge model with {n_features} features...")
        ridge_results, ridge_overall_r2 = train_and_evaluate_enhanced(data, model_type='ridge', n_features=n_features)
        ridge_analysis = analyze_results(ridge_results, f"Enhanced Ridge ({n_features} features)", ridge_overall_r2)
        
        # Save results
        lasso_results.to_csv(f"stock_ret_predictions/enhanced_lasso_results_{n_features}features.csv", index=False)
        ridge_results.to_csv(f"stock_ret_predictions/enhanced_ridge_results_{n_features}features.csv", index=False)
        
        # Calculate MSE for summary (avoid re-reading CSV)
        lasso_mse = np.nan
        ridge_mse = np.nan
        lasso_obs = 0
        ridge_obs = 0
        
        if lasso_overall_r2 is not None:
            lasso_pred_data = pd.read_csv(f"stock_ret_predictions/stock_predictions_lasso.csv")
            lasso_mse = mean_squared_error(lasso_pred_data['stock_exret'], lasso_pred_data['lasso'])
            lasso_obs = len(lasso_pred_data)
            
        if ridge_overall_r2 is not None:
            ridge_pred_data = pd.read_csv(f"stock_ret_predictions/stock_predictions_ridge.csv")
            ridge_mse = mean_squared_error(ridge_pred_data['stock_exret'], ridge_pred_data['ridge'])
            ridge_obs = len(ridge_pred_data)
        
        # Store overall model performance for summary
        model_summary.append({
            'model': 'Lasso',
            'n_features': n_features,
            'overall_r2': lasso_overall_r2 if lasso_overall_r2 is not None else np.nan,
            'overall_mse': lasso_mse,
            'mean_period_r2': lasso_results['test_r2'].mean(),
            'median_period_r2': lasso_results['test_r2'].median(),
            'std_period_r2': lasso_results['test_r2'].std(),
            'min_period_r2': lasso_results['test_r2'].min(),
            'max_period_r2': lasso_results['test_r2'].max(),
            'positive_r2_periods': (lasso_results['test_r2'] > 0).sum(),
            'total_periods': len(lasso_results),
            'target_achieved_periods': (lasso_results['test_r2'] >= 0.01).sum(),
            'mean_alpha': lasso_results['best_alpha'].mean(),
            'total_test_observations': lasso_obs
        })
        
        model_summary.append({
            'model': 'Ridge',
            'n_features': n_features,
            'overall_r2': ridge_overall_r2 if ridge_overall_r2 is not None else np.nan,
            'overall_mse': ridge_mse,
            'mean_period_r2': ridge_results['test_r2'].mean(),
            'median_period_r2': ridge_results['test_r2'].median(),
            'std_period_r2': ridge_results['test_r2'].std(),
            'min_period_r2': ridge_results['test_r2'].min(),
            'max_period_r2': ridge_results['test_r2'].max(),
            'positive_r2_periods': (ridge_results['test_r2'] > 0).sum(),
            'total_periods': len(ridge_results),
            'target_achieved_periods': (ridge_results['test_r2'] >= 0.01).sum(),
            'mean_alpha': ridge_results['best_alpha'].mean(),
            'total_test_observations': ridge_obs
        })
        
        # Check if we achieved our target
        lasso_target_achieved = (lasso_results['test_r2'] >= 0.01).sum()
        ridge_target_achieved = (ridge_results['test_r2'] >= 0.01).sum()
        
        print(f"\n🎯 SUMMARY FOR {n_features} FEATURES:")
        if lasso_overall_r2 is not None:
            print(f"  Lasso Overall R²: {lasso_overall_r2:.6f}")
        else:
            print(f"  Lasso Overall R²: N/A (no predictions)")
        if ridge_overall_r2 is not None:
            print(f"  Ridge Overall R²: {ridge_overall_r2:.6f}")
        else:
            print(f"  Ridge Overall R²: N/A (no predictions)")
        print(f"  Lasso periods with R² ≥ 0.01: {lasso_target_achieved}/{len(lasso_results)}")
        print(f"  Ridge periods with R² ≥ 0.01: {ridge_target_achieved}/{len(ridge_results)}")
        
        if (lasso_overall_r2 and lasso_overall_r2 > 0) or (ridge_overall_r2 and ridge_overall_r2 > 0):
            print("🎉 SUCCESS! We achieved positive overall out-of-sample R²!")
        if lasso_target_achieved > 0 or ridge_target_achieved > 0:
            print("🎉 SUCCESS! We achieved R² ≥ 0.01 in some periods!")
    
    # Save overall model performance summary
    summary_df = pd.DataFrame(model_summary)
    
    # Round numerical columns for better readability
    numerical_cols = ['overall_r2', 'overall_mse', 'mean_period_r2', 'median_period_r2', 
                     'std_period_r2', 'min_period_r2', 'max_period_r2', 'mean_alpha']
    summary_df[numerical_cols] = summary_df[numerical_cols].round(6)
    
    # Add percentage columns for easier interpretation
    summary_df['positive_r2_percentage'] = (summary_df['positive_r2_periods'] / summary_df['total_periods'] * 100).round(1)
    summary_df['target_achieved_percentage'] = (summary_df['target_achieved_periods'] / summary_df['total_periods'] * 100).round(1)
    
    # Reorder columns for better presentation
    column_order = [
        'model', 'n_features', 'overall_r2', 'overall_mse', 
        'mean_period_r2', 'median_period_r2', 'std_period_r2', 'min_period_r2', 'max_period_r2',
        'positive_r2_periods', 'positive_r2_percentage', 'target_achieved_periods', 'target_achieved_percentage',
        'total_periods', 'total_test_observations', 'mean_alpha'
    ]
    summary_df = summary_df[column_order]
    
    # Save to CSV with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"stock_ret_predictions/model_performance_summary.csv"
    summary_df.to_csv(summary_filename, index=False)
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("All model predictions saved! Check the stock_ret_predictions/ folder to analyze stock return prediction performance.")
    print(f"\n📊 MODEL PERFORMANCE SUMMARY SAVED: {summary_filename}")
    print("\nSummary Table:")
    print("="*100)
    print(summary_df.to_string(index=False))
    
    # Find and highlight best performing models
    print(f"\n🏆 BEST PERFORMING MODELS:")
    print("="*50)
    best_overall_r2 = summary_df.loc[summary_df['overall_r2'].idxmax()]
    best_mean_period_r2 = summary_df.loc[summary_df['mean_period_r2'].idxmax()]
    
    print(f"Best Overall R²: {best_overall_r2['model']} with {best_overall_r2['n_features']} features (R² = {best_overall_r2['overall_r2']:.6f})")
    print(f"Best Mean Period R²: {best_mean_period_r2['model']} with {best_mean_period_r2['n_features']} features (R² = {best_mean_period_r2['mean_period_r2']:.6f})")
    
    # Show models that achieved target
    successful_models = summary_df[summary_df['target_achieved_periods'] > 0]
    if len(successful_models) > 0:
        print(f"\n✅ Models achieving R² ≥ 0.01 in some periods:")
        for _, row in successful_models.iterrows():
            print(f"  {row['model']} ({row['n_features']} features): {row['target_achieved_periods']}/{row['total_periods']} periods ({row['target_achieved_percentage']:.1f}%)")
    
    # 🎯 DEFINE AND SELECT THE BEST MODEL
    print(f"\n" + "="*80)
    print("🎯 BEST MODEL SELECTION FOR PORTFOLIO ANALYSIS")
    print("="*80)
    
    # Filter out models with NaN overall_r2
    valid_models = summary_df.dropna(subset=['overall_r2'])
    
    if len(valid_models) == 0:
        print("❌ ERROR: No valid models found!")
        best_model_info = None
    else:
        # Define selection criteria (you can modify these)
        print("📋 SELECTION CRITERIA:")
        print("  Primary: Highest Overall R² (most important for portfolio performance)")
        print("  Secondary: Consistency (positive R² in multiple periods)")
        print("  Tertiary: Target achievement (R² ≥ 0.01 in some periods)")
        
        # Primary criterion: Best overall R²
        best_model_row = valid_models.loc[valid_models['overall_r2'].idxmax()]
        
        # Create best model info
        best_model_info = {
            'model_type': best_model_row['model'].lower(),  # 'lasso' or 'ridge'
            'n_features': int(best_model_row['n_features']),
            'overall_r2': best_model_row['overall_r2'],
            'overall_mse': best_model_row['overall_mse'],
            'mean_period_r2': best_model_row['mean_period_r2'],
            'positive_r2_periods': int(best_model_row['positive_r2_periods']),
            'target_achieved_periods': int(best_model_row['target_achieved_periods']),
            'total_periods': int(best_model_row['total_periods']),
            'predictions_file': f"stock_ret_predictions/stock_predictions_{best_model_row['model'].lower()}.csv"
        }
        
        print(f"\n🏆 SELECTED BEST MODEL:")
        print(f"  Model: {best_model_info['model_type'].title()}")
        print(f"  Features: {best_model_info['n_features']}")
        print(f"  Overall R²: {best_model_info['overall_r2']:.6f}")
        print(f"  Mean Period R²: {best_model_info['mean_period_r2']:.6f}")
        print(f"  Positive R² Periods: {best_model_info['positive_r2_periods']}/{best_model_info['total_periods']}")
        print(f"  Target Achieved: {best_model_info['target_achieved_periods']}/{best_model_info['total_periods']} periods")
        print(f"  Predictions File: {best_model_info['predictions_file']}")
        
        # Save best model info for portfolio analysis
        import json
        with open('stock_ret_predictions/best_model_info.json', 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        print(f"\n💾 Best model info saved to: stock_ret_predictions/best_model_info.json")
        print("   This file will be used by portfolio_analysis_detailed.py")
        
        # Alternative models ranking
        print(f"\n📊 ALL MODELS RANKED BY OVERALL R²:")
        ranked_models = valid_models.sort_values('overall_r2', ascending=False)
        for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
            marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"  {marker} {row['model']} ({row['n_features']} features): R² = {row['overall_r2']:.6f}")
        
        # Recommendation for portfolio analysis
        if best_model_info['overall_r2'] > 0:
            print(f"\n✅ RECOMMENDATION: Use {best_model_info['model_type'].title()} model for portfolio analysis")
            print(f"   Expected out-of-sample performance: R² = {best_model_info['overall_r2']:.6f}")
        else:
            print(f"\n⚠️  WARNING: Best model has negative R² ({best_model_info['overall_r2']:.6f})")
            print("   Consider improving features or trying different models before portfolio analysis") 