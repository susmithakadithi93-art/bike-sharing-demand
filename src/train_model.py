import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Set plot aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

def load_data(filepath):
    """Load Bike Sharing dataset and verify essential structure."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file missing at: {filepath}")
    
    df = pd.read_csv(filepath)
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df

def feature_engineering(df):
    """
    Perform targeted feature transformations:
    - Extract calendar properties from date
    - Create cyclical encodings for hour and month
    - Add weather/temperature interaction features
    """
    data = df.copy()
    
    # Date-based features
    data['day_of_month'] = data['dteday'].dt.day
    data['is_weekend'] = ((data['workingday'] == 0) & (data['holiday'] == 0)).astype(int)
    
    # Cyclical hour encoding
    data['hr_sin'] = np.sin(2 * np.pi * data['hr'] / 24.0)
    data['hr_cos'] = np.cos(2 * np.pi * data['hr'] / 24.0)
    
    # Cyclical month encoding
    data['mnth_sin'] = np.sin(2 * np.pi * data['mnth'] / 12.0)
    data['mnth_cos'] = np.cos(2 * np.pi * data['mnth'] / 12.0)
    
    # Physical interaction terms
    data['temp_hum_ratio'] = data['temp'] / (data['hum'] + 0.01)
    data['atemp_wind_prod'] = data['atemp'] * data['windspeed']
    
    return data

def build_preprocessor():
    """
    Construct Scikit-learn ColumnTransformer for numerical scaling 
    and categorical encoding.
    """
    numeric_features = [
        'temp', 'atemp', 'hum', 'windspeed', 'hr', 'mnth', 'yr', 
        'weekday', 'day_of_month', 'hr_sin', 'hr_cos', 'mnth_sin', 
        'mnth_cos', 'temp_hum_ratio', 'atemp_wind_prod'
    ]
    categorical_features = ['season', 'weathersit', 'holiday', 'workingday', 'is_weekend']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor

def train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train baseline and ensemble models, compute exact performance metrics,
    and select the best performing regressor.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    }
    
    results = {}
    fitted_pipelines = {}
    
    print("=" * 65)
    print(f"{'Model Name':<22} | {'MAE':<10} | {'RMSE':<10} | {'R² Score':<10}")
    print("=" * 65)
    
    best_model_name = None
    best_r2 = -float('inf')
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Enforce non-negative demand predictions
        y_pred_clipped = np.clip(y_pred, 0, None)
        
        mae = mean_absolute_error(y_test, y_pred_clipped)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
        r2 = r2_score(y_test, y_pred_clipped)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'predictions': y_pred_clipped}
        fitted_pipelines[name] = pipeline
        
        print(f"{name:<22} | {mae:<10.3f} | {rmse:<10.3f} | {r2:<10.4f}")
        
    print("=" * 65)
    
    # Check for ties or near-ties in model performance (R2 threshold 0.0005)
    sorted_models = sorted(results.items(), key=lambda item: item[1]['R2'], reverse=True)
    top_model_name, top_metrics = sorted_models[0]
    second_model_name, second_metrics = sorted_models[1]
    
    if abs(top_metrics['R2'] - second_metrics['R2']) < 0.0005:
        print(f"TIE DETECTED between '{top_model_name}' (R²={top_metrics['R2']:.4f}) and '{second_model_name}' (R²={second_metrics['R2']:.4f}).")
        # Tie-breaker: prefer lower RMSE, then lower MAE, then simpler model
        if top_metrics['RMSE'] <= second_metrics['RMSE']:
            best_model_name = top_model_name
            print(f"Tie-breaker decision: Selected '{best_model_name}' due to lower/equal RMSE ({top_metrics['RMSE']:.3f} vs {second_metrics['RMSE']:.3f}).")
        else:
            best_model_name = second_model_name
            print(f"Tie-breaker decision: Selected '{best_model_name}' due to lower RMSE ({second_metrics['RMSE']:.3f} vs {top_metrics['RMSE']:.3f}).")
    else:
        best_model_name = top_model_name
        print(f"Selected Best Model: {best_model_name} (R² = {top_metrics['R2']:.4f}, RMSE = {top_metrics['RMSE']:.3f}) - No tie.")

    # Feature Importance / Coefficient Analysis:
    # NOTE ON DIRECTIONALITY (Known Trap Prevention):
    # - Linear / Ridge coefficients carry direction (+ / -).
    # - Tree model feature_importances_ are ALWAYS POSITIVE magnitudes and carry NO directional sign.
    best_pipe = fitted_pipelines[best_model_name]
    regressor = best_pipe.named_steps['regressor']
    
    if hasattr(regressor, 'coef_'):
        print(f"\nModel '{best_model_name}' Linear Coefficients (Directional):")
        # Extract feature names from preprocessor
        cat_encoder = best_pipe.named_steps['preprocessor'].named_transformers_['cat']
        cat_names = cat_encoder.get_feature_names_out(['season', 'weathersit', 'holiday', 'workingday', 'is_weekend']).tolist()
        num_names = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'mnth', 'yr', 'weekday', 'day_of_month', 'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'temp_hum_ratio', 'atemp_wind_prod']
        all_feature_names = num_names + cat_names
        
        coefs = pd.Series(regressor.coef_, index=all_feature_names).sort_values()
        print("  Top negative drivers (decreases demand):", coefs.head(3).to_dict())
        print("  Top positive drivers (increases demand):", coefs.tail(3).to_dict())
    elif hasattr(regressor, 'feature_importances_'):
        print(f"\nModel '{best_model_name}' Feature Importances (MAGNITUDE ONLY - NON-DIRECTIONAL):")
        print("  Note: Tree feature importances are always positive and carry no direction. Direction is evaluated via linear model coefficients.")
        cat_encoder = best_pipe.named_steps['preprocessor'].named_transformers_['cat']
        cat_names = cat_encoder.get_feature_names_out(['season', 'weathersit', 'holiday', 'workingday', 'is_weekend']).tolist()
        num_names = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'mnth', 'yr', 'weekday', 'day_of_month', 'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'temp_hum_ratio', 'atemp_wind_prod']
        all_feature_names = num_names + cat_names
        
        importances = pd.Series(regressor.feature_importances_, index=all_feature_names).sort_values(ascending=False)
        print("  Top feature importance magnitudes:", importances.head(5).to_dict())
    
    return best_model_name, fitted_pipelines, results


def generate_visualizations(df, test_df, results, best_model_name, output_dir):
    """Generate high-resolution chart images for BI report and README."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Hourly Demand Pattern by Working Day
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.lineplot(
        data=df, x='hr', y='cnt', hue='workingday', 
        palette={0: '#e74c3c', 1: '#2980b9'}, ax=ax, linewidth=2.5
    )
    ax.set_title('Hourly Bike Demand: Workdays vs Non-Workdays', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Hour of Day (0 - 23)', fontsize=11)
    ax.set_ylabel('Average Rental Count (bikes/hr)', fontsize=11)
    ax.legend(['Weekend / Holiday', 'Workday'], title='Day Category', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_hourly_demand_patterns.png'))
    plt.close()
    
    # 2. Seasonal and Weather Condition Impact
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    df_plot = df.copy()
    df_plot['season_label'] = df_plot['season'].map(season_labels)
    
    sns.barplot(data=df_plot, x='season_label', y='cnt', palette='Blues_d', ax=ax1, ci=None)
    ax1.set_title('Average Hourly Demand by Season', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Season', fontsize=11)
    ax1.set_ylabel('Mean Bike Count', fontsize=11)
    
    weather_labels = {1: 'Clear/Few Clouds', 2: 'Mist/Cloudy', 3: 'Light Snow/Rain', 4: 'Heavy Rain/Ice'}
    df_plot['weather_label'] = df_plot['weathersit'].map(weather_labels)
    
    sns.barplot(data=df_plot, x='weather_label', y='cnt', palette='Reds_d', ax=ax2, ci=None)
    ax2.set_title('Average Hourly Demand by Weather Severity', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Weather Condition', fontsize=11)
    ax2.set_ylabel('Mean Bike Count', fontsize=11)
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_seasonal_weather_demand.png'))
    plt.close()
    
    # 3. Actual vs Predicted Chronological Sample (Last 200 hours of test set)
    y_test = test_df['cnt'].values
    y_pred = results[best_model_name]['predictions']
    
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.plot(y_test[-200:], label='Actual Hourly Demand', color='#2c3e50', linewidth=1.8, alpha=0.85)
    ax.plot(y_pred[-200:], label=f'Predicted ({best_model_name})', color='#e67e22', linestyle='--', linewidth=1.8)
    ax.set_title(f'Actual vs Predicted Demand Forecast (Chronological 200-Hour Window)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Chronological Hours (Test Horizon)', fontsize=11)
    ax.set_ylabel('Bike Rental Count', fontsize=11)
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_model_actual_vs_predicted.png'))
    plt.close()
    
    # 4. Residual Error Distribution
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.histplot(residuals, kde=True, color='#8e44ad', ax=ax, bins=40)
    ax.set_title(f'Prediction Residual Distribution ({best_model_name})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Residual Error (Actual - Predicted)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.axvline(0, color='red', linestyle=':', linewidth=1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_residuals_distribution.png'))
    plt.close()

    # 5. Fleet Rebalancing Demand Heatmap (Hour vs Weekday)
    pivot_demand = test_df.copy()
    pivot_demand['pred_cnt'] = y_pred
    heatmap_data = pivot_demand.pivot_table(index='hr', columns='weekday', values='pred_cnt', aggfunc='mean')
    
    weekday_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    heatmap_data.columns = [weekday_names[c] for c in heatmap_data.columns]
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Predicted Bikes/Hour'}, ax=ax)
    ax.set_title('Operational Rebalancing Matrix: Average Predicted Demand', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Day of Week', fontsize=11)
    ax.set_ylabel('Hour of Day (24-Hour Format)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_rebalancing_schedule_heatmap.png'))
    plt.close()
    
    print(f"Successfully generated 5 chart images in '{output_dir}'.")

def main():
    base_dir = r"D:\Data analytics\Susi\bike-sharing-demand"
    data_path = os.path.join(base_dir, "data", "bike_sharing_hour.csv")
    model_path = os.path.join(base_dir, "models", "demand_model.pkl")
    forecast_path = os.path.join(base_dir, "results", "hourly_forecast.csv")
    figures_dir = os.path.join(base_dir, "results", "figures")
    
    print("Step 1: Loading Dataset...")
    raw_df = load_data(data_path)
    print(f"Dataset shape: {raw_df.shape}")
    
    print("\nStep 2: Feature Engineering & Data Preparation...")
    df_engineered = feature_engineering(raw_df)
    
    # CRITICAL LEAKAGE PREVENTION: Drop target components casual & registered + instant
    leakage_cols = ['casual', 'registered', 'instant']
    features_df = df_engineered.drop(columns=leakage_cols)
    print(f"Dropped leakage columns: {leakage_cols}")
    
    # Chronological Split (80% train, 20% test) to prevent time-series look-ahead bias
    split_idx = int(len(features_df) * 0.8)
    train_data = features_df.iloc[:split_idx].copy()
    test_data = features_df.iloc[split_idx:].copy()
    
    drop_from_X = ['dteday', 'cnt']
    X_train = train_data.drop(columns=drop_from_X)
    y_train = train_data['cnt']
    X_test = test_data.drop(columns=drop_from_X)
    y_test = test_data['cnt']
    
    print(f"Train split size: {len(X_train)} samples ({train_data['dteday'].min().date()} to {train_data['dteday'].max().date()})")
    print(f"Test split size:  {len(X_test)} samples ({test_data['dteday'].min().date()} to {test_data['dteday'].max().date()})")
    
    print("\nStep 3: Building Preprocessor & Training Models...")
    preprocessor = build_preprocessor()
    best_name, fitted_pipelines, results = train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor)
    
    best_pipeline = fitted_pipelines[best_name]
    
    print(f"\nStep 4: Saving Best Model to '{model_path}'...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_pipeline, model_path, compress=3)
    print("Model pipeline successfully serialized.")
    
    print("\nStep 5: Exporting Managerial Forecast Table...")
    best_predictions = results[best_name]['predictions']
    forecast_df = test_data[['dteday', 'hr', 'workingday', 'weathersit', 'temp', 'cnt']].copy()
    forecast_df.rename(columns={'cnt': 'actual_demand'}, inplace=True)
    forecast_df['predicted_demand'] = best_predictions.round(1)
    forecast_df['prediction_error'] = forecast_df['actual_demand'] - forecast_df['predicted_demand']
    forecast_df['abs_error'] = forecast_df['prediction_error'].abs()
    
    # Add operational rebalancing alert category based on predicted volume
    def assign_action_flag(pred):
        if pred >= 400:
            return 'CRITICAL_REBALANCE_HIGH'
        elif pred >= 250:
            return 'MODERATE_REBALANCE'
        elif pred <= 30:
            return 'LOW_FLEET_IDLE'
        else:
            return 'STANDARD_MONITOR'
            
    forecast_df['rebalancing_action'] = forecast_df['predicted_demand'].apply(assign_action_flag)
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Exported manager forecast table with {len(forecast_df)} hourly rows to '{forecast_path}'.")
    
    print("\nStep 6: Generating Visualization Visuals...")
    generate_visualizations(df_engineered, test_data, results, best_name, figures_dir)
    
    print("\nWorkflow Execution Complete Successfully!")

if __name__ == "__main__":
    main()
