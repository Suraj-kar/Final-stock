import numpy as np
import pandas as pd
import os
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from Regression import MultiLinearRegression, DataScaler

# Get the project root directory (where manage.py is located)
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets"
MODEL_PATH = PROJECT_ROOT / "model" / "saved_models"

def clean_data(data):
    """Clean the input data and extract features."""
    if 'Vol' in data.columns and isinstance(data['Vol'].iloc[0], str):
        data['Vol'] = data['Vol'].str.replace(',', '').astype(float)
    X = data[['Open', 'High', 'Low', 'Vol']].values
    return X

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    rmsae = np.sqrt(mean_absolute_error(y_true, y_pred))
    return mse, rmse, r2, rmsae

def kfold_cross_validation(X, Y, company_name, n_splits=5):
    """Perform k-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_metrics = []
    best_model = None
    best_scaler = None
    best_r2 = -float('inf')
    
    print(f"\n=== {n_splits}-Fold Cross-Validation for {company_name} ===")
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        # Scale the data
        scaler = DataScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = MultiLinearRegression(learning_rate=0.01, num_iterations=1000)
        model.fit(X_train_scaled, Y_train)
        
        # Evaluate on test fold
        Y_test_pred = model.predict(X_test_scaled)
        mse, rmse, r2, rmsae = calculate_metrics(Y_test, Y_test_pred)
        
        fold_metrics.append((mse, rmse, r2, rmsae))
        print(f"Fold {fold+1}: MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, RMSAE: {rmsae:.4f}")
        
        # Keep track of the best model based on R²
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_scaler = scaler
    
    # Calculate average metrics across all folds
    avg_mse = np.mean([m[0] for m in fold_metrics])
    avg_rmse = np.mean([m[1] for m in fold_metrics])
    avg_r2 = np.mean([m[2] for m in fold_metrics])
    avg_rmsae = np.mean([m[3] for m in fold_metrics])
    
    print(f"\nAverage Cross-Validation Metrics for {company_name}:")
    print(f"MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, R²: {avg_r2:.4f}, RMSAE: {avg_rmsae:.4f}")
    
    # Final training on full dataset for deployment
    print(f"\nTraining final model on full dataset...")
    
    # Scale the full data
    final_scaler = DataScaler()
    X_scaled = final_scaler.fit_transform(X)
    
    # Train final model
    final_model = MultiLinearRegression(learning_rate=0.01, num_iterations=1000)
    final_model.fit(X_scaled, Y)
    
    return final_scaler, final_model, (avg_mse, avg_rmse, avg_r2, avg_rmsae)

def train_test_evaluation(X, Y, company_name):
    """Perform train-test split evaluation."""
    split_index = int(0.8 * len(X))
    X_train, Y_train = X[:split_index], Y[:split_index]
    X_test, Y_test = X[split_index:], Y[split_index:]
    
    scaler = DataScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = MultiLinearRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train_scaled, Y_train)
    
    Y_test_pred = model.predict(X_test_scaled)
    mse, rmse, r2, rmsae = calculate_metrics(Y_test, Y_test_pred)
    
    print(f"\n=== Train-Test Evaluation for {company_name} ===")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, RMSAE: {rmsae:.4f}")
    return scaler, model

def train_company_model(company_file, use_kfold=True):
    """Train and save model for a single company."""
    file_path = DATASET_PATH / company_file
    print(f"Reading data from: {file_path}")
    
    df_train = pd.read_csv(file_path)
    company_name = company_file.split('.')[0]
    X = clean_data(df_train)
    Y = df_train["Close"].values
    
    print(f"\n=== Training {company_name} ===")
    
    # Perform evaluations
    if use_kfold:
        scaler, model, metrics = kfold_cross_validation(X, Y, company_name)
    else:
        scaler, model = train_test_evaluation(X, Y, company_name)
    
    # Create saved_models directory if it doesn't exist
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save models
    scaler_path = MODEL_PATH / f"{company_name}_scaler.pkl"
    model_path = MODEL_PATH / f"{company_name}_model.pkl"
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)
    
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved model to: {model_path}")
    
    return scaler, model

def main():
    """Main function to train models for all companies."""
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Looking for datasets in: {DATASET_PATH}")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {DATASET_PATH}")
    
    company_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.csv')]
    
    if not company_files:
        raise FileNotFoundError(f"No CSV files found in {DATASET_PATH}")
    
    print(f"\nFound {len(company_files)} company files:")
    for file in company_files:
        print(f"- {file}")
    
    # Ask user whether to use K-fold cross-validation
    use_kfold = input("\nUse K-fold cross-validation? (y/n): ").lower().startswith('y')
    
    for company_file in company_files:
        train_company_model(company_file, use_kfold=use_kfold)

if __name__ == "__main__":
    main()