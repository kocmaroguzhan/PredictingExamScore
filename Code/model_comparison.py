import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the labeled data"""
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_folder = os.path.join(parent_folder, "Dataset")
    
    # Load data
    df = pd.read_csv(os.path.join(dataset_folder, "labeled_data.csv"))
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['UID', 'MidtermClass']]
    X = df[feature_cols].values
    y = df['MidtermClass'].values
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Target range: {y.min()} - {y.max()}")
    print(f"📈 Features: {len(feature_cols)}")
    print(f"📋 Feature names: {feature_cols}")
    
    return X, y, feature_cols

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, scaler=None):
    """Evaluate a single model and return metrics"""
    # Scale features if scaler is provided
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()
    
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'predictions': y_pred,
        'actual': y_test
    }

def run_model_comparison():
    """Run comprehensive model comparison"""
    print("🚀 Starting Model Comparison...")
    
    # Load data
    X, y, feature_cols = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"📚 Training set: {X_train.shape[0]} samples")
    print(f"🧪 Test set: {X_test.shape[0]} samples")
    
    # Define models to test
    models = {
        # Linear Models
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        
        # Tree-based Models
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        
        # Neighbor-based Models
        'K-Neighbors (k=3)': KNeighborsRegressor(n_neighbors=3),
        'K-Neighbors (k=5)': KNeighborsRegressor(n_neighbors=5),
        'K-Neighbors (k=7)': KNeighborsRegressor(n_neighbors=7),
        
        # Neural Network
        'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        
        # Support Vector Machine
        'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'SVR (Linear)': SVR(kernel='linear', C=1.0),
    }
    
    # Scalers
    scalers = {
        'No Scaling': None,
        'Standard Scaler': StandardScaler(),
        'Robust Scaler': RobustScaler()
    }
    
    results = []
    
    # Test each model with different scalers
    for scaler_name, scaler in scalers.items():
        print(f"\n🔧 Testing with {scaler_name}...")
        
        for model_name, model in models.items():
            try:
                result = evaluate_model(model, X_train, X_test, y_train, y_test, 
                                     f"{model_name} ({scaler_name})", scaler)
                results.append(result)
                
                print(f"✅ {model_name} ({scaler_name}): R² = {result['r2']:.3f}, RMSE = {result['rmse']:.3f}")
                
            except Exception as e:
                print(f"❌ {model_name} ({scaler_name}): Error - {str(e)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by R² score
    results_df = results_df.sort_values('r2', ascending=False)
    
    # Print top 10 results
    print("\n🏆 TOP 10 MODELS (by R² score):")
    print("=" * 80)
    for idx, row in results_df.head(10).iterrows():
        print(f"{idx+1:2d}. {row['model_name']:<40} R²: {row['r2']:.3f} | RMSE: {row['rmse']:.3f} | MAE: {row['mae']:.3f}")
    
    # Save detailed results
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_folder = os.path.join(parent_folder, "Dataset")
    results_path = os.path.join(dataset_folder, "model_comparison_results.csv")
    
    # Save results without the large prediction arrays
    save_df = results_df.drop(['predictions', 'actual'], axis=1)
    save_df.to_csv(results_path, index=False)
    print(f"\n💾 Detailed results saved to: {results_path}")
    
    # Feature importance analysis for tree-based models
    print("\n🌳 FEATURE IMPORTANCE ANALYSIS:")
    print("=" * 50)
    
    tree_models = ['Random Forest', 'Extra Trees', 'Gradient Boosting']
    for model_name in tree_models:
        model_results = results_df[results_df['model_name'].str.contains(model_name)]
        if not model_results.empty:
            best_model_result = model_results.iloc[0]
            print(f"\n📊 {model_name} - Best R²: {best_model_result['r2']:.3f}")
            
            # Get the actual model to extract feature importance
            for name, model in models.items():
                if model_name in name and 'No Scaling' in name:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_importance_df = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        print("Top 5 most important features:")
                        for idx, row in feature_importance_df.head().iterrows():
                            print(f"  {row['feature']}: {row['importance']:.4f}")
                    break
    
    return results_df

def analyze_predictions(results_df):
    """Analyze prediction patterns"""
    print("\n📈 PREDICTION ANALYSIS:")
    print("=" * 50)
    
    # Get best model
    best_result = results_df.iloc[0]
    predictions = best_result['predictions']
    actual = best_result['actual']
    
    print(f"🏆 Best Model: {best_result['model_name']}")
    print(f"📊 R² Score: {best_result['r2']:.3f}")
    print(f"📏 RMSE: {best_result['rmse']:.3f}")
    
    # Prediction vs Actual analysis
    errors = np.abs(predictions - actual)
    print(f"📉 Mean Absolute Error: {np.mean(errors):.2f}")
    print(f"📊 Error Std Dev: {np.std(errors):.2f}")
    print(f"🎯 Perfect Predictions: {np.sum(errors == 0)}/{len(errors)} ({np.sum(errors == 0)/len(errors)*100:.1f}%)")
    
    # Error distribution
    print(f"\n📊 Error Distribution:")
    print(f"  Small errors (≤1): {np.sum(errors <= 1)} ({np.sum(errors <= 1)/len(errors)*100:.1f}%)")
    print(f"  Medium errors (2-3): {np.sum((errors > 1) & (errors <= 3))} ({(np.sum((errors > 1) & (errors <= 3))/len(errors)*100):.1f}%)")
    print(f"  Large errors (>3): {np.sum(errors > 3)} ({np.sum(errors > 3)/len(errors)*100:.1f}%)")
    
    # Worst predictions
    worst_indices = np.argsort(errors)[-5:]
    print(f"\n❌ Worst Predictions:")
    for idx in worst_indices:
        print(f"  Actual: {actual[idx]}, Predicted: {predictions[idx]:.1f}, Error: {errors[idx]:.1f}")

if __name__ == "__main__":
    # Run model comparison
    results = run_model_comparison()
    
    # Analyze predictions
    analyze_predictions(results)
    
    print("\n🎉 Model comparison completed!") 