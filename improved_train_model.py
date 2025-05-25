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
        test_predictions = pd.DataFrame({
            'permno': test_data['permno'].values,
            'date': test_data['date'].values,
            'year': test_data['date'].dt.year.values,
            'month': test_data['date'].dt.month.values,
            'stock_exret': y_test.values,  # actual returns
            model_type: test_pred  # predicted returns for this model
        })
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
        predictions_filename = f"stock_predictions_{model_type}.csv"
        all_predictions_df.to_csv(predictions_filename, index=False)
        print(f"Individual predictions saved to: {predictions_filename}")
        print(f"Predictions shape: {all_predictions_df.shape}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def analyze_results(results_df, model_name):
    """
    Analyze the results and provide insights.
    """
    print(f"\n{model_name} Results Analysis:")
    print("="*50)
    
    # Basic statistics
    test_r2_mean = results_df['test_r2'].mean()
    test_r2_std = results_df['test_r2'].std()
    test_r2_median = results_df['test_r2'].median()
    positive_r2_count = (results_df['test_r2'] > 0).sum()
    target_r2_count = (results_df['test_r2'] >= 0.01).sum()
    
    print(f"Test R² Statistics:")
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
    
    for n_features in feature_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_features} features")
        print(f"{'='*60}")
        
        # Train and evaluate enhanced Lasso model
        print(f"\nTraining Enhanced Lasso model with {n_features} features...")
        lasso_results = train_and_evaluate_enhanced(data, model_type='lasso', n_features=n_features)
        lasso_analysis = analyze_results(lasso_results, f"Enhanced Lasso ({n_features} features)")
        
        # Train and evaluate enhanced Ridge model
        print(f"\nTraining Enhanced Ridge model with {n_features} features...")
        ridge_results = train_and_evaluate_enhanced(data, model_type='ridge', n_features=n_features)
        ridge_analysis = analyze_results(ridge_results, f"Enhanced Ridge ({n_features} features)")
        
        # Save results
        lasso_results.to_csv(f"enhanced_lasso_results_{n_features}features.csv", index=False)
        ridge_results.to_csv(f"enhanced_ridge_results_{n_features}features.csv", index=False)
        
        # Check if we achieved our target
        lasso_target_achieved = (lasso_results['test_r2'] >= 0.01).sum()
        ridge_target_achieved = (ridge_results['test_r2'] >= 0.01).sum()
        
        print(f"\nTarget Achievement (R² ≥ 0.01):")
        print(f"  Lasso: {lasso_target_achieved}/{len(lasso_results)} periods")
        print(f"  Ridge: {ridge_target_achieved}/{len(ridge_results)} periods")
        
        if lasso_target_achieved > 0 or ridge_target_achieved > 0:
            print("🎉 SUCCESS! We achieved positive R² ≥ 0.01 in some periods!")
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("All model predictions saved! Check the stock_predictions_*.csv files to choose the best model for portfolio analysis.") 