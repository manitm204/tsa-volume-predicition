import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
import matplotlib.pyplot as plt

DATE_COL = "Date"
Y_COL = "Volume"

def thanksgiving_date(year):
    """
    Determine Thanksgivings Date
    """
    nov = pd.Timestamp(year=year, month=11, day=1)
    first_thursday = nov + pd.offsets.Week(weekday=3)
    return first_thursday + pd.offsets.Week(3)


def add_holiday_week_features(df):
    """
    Add columns to the dataframe for days to thanksgiving and christmas, 
    and is thanksgiving and christmas week. 
    """
    df = df.copy()
    dates = df[DATE_COL]

    # Thanksgiving
    thanksgiving = dates.dt.year.apply(thanksgiving_date)
    df["days_to_thanksgiving"] = (thanksgiving - dates).dt.days

    # Clip to window
    df["days_to_thanksgiving"] = df["days_to_thanksgiving"].clip(-21, 21)

    # Thanksgiving week
    df["is_thanksgiving_week"] = (
        (df["days_to_thanksgiving"] >= -3) &
        (df["days_to_thanksgiving"] <= 3)
    ).astype(int)

    # Christmas
    christmas = pd.to_datetime(
        df[DATE_COL].dt.year.astype(str) + "-12-25"
    )
    df["days_to_christmas"] = (christmas - dates).dt.days
    df["days_to_christmas"] = df["days_to_christmas"].clip(-30, 30)

    # Christmas week
    df["is_christmas_week"] = (
        (df["days_to_christmas"] >= -5) &
        (df["days_to_christmas"] <= 2)
    ).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time features that will be fed into the XGBoost Model
    """
    min_year = df[DATE_COL].dt.year.min()
    max_year = df[DATE_COL].dt.year.max()
    us_holidays = holidays.US(years=range(min_year, max_year + 1))
    
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["month"] = df[DATE_COL].dt.month
    df["day_of_month"] = df[DATE_COL].dt.day
    df["quarter"] = df[DATE_COL].dt.quarter
    df["day_of_year"] = df[DATE_COL].dt.dayofyear
    
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    
    df["is_holiday"] = df[DATE_COL].dt.date.apply(lambda x: int(x in us_holidays))
    
    hol_dates = pd.to_datetime(sorted(us_holidays.keys()))
    
    date_vals = df[DATE_COL].values.astype("datetime64[D]")
    hol_vals = hol_dates.values.astype("datetime64[D]")
    
    idx_next = np.searchsorted(hol_vals, date_vals, side="left")
    idx_prev = idx_next - 1
    
    next_h = np.where(idx_next < len(hol_vals), hol_vals[idx_next], np.datetime64("NaT"))
    prev_h = np.where(idx_prev >= 0, hol_vals[idx_prev], np.datetime64("NaT"))
    
    df["days_to_holiday"] = (next_h - date_vals).astype("timedelta64[D]").astype("float")
    df["days_since_holiday"] = (date_vals - prev_h).astype("timedelta64[D]").astype("float")
    
    df["days_to_holiday"] = df["days_to_holiday"].fillna(9999)
    df["days_since_holiday"] = df["days_since_holiday"].fillna(9999)

    
    return df

def add_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features to dataframe
    """
    # Lags
    for L in [1, 2, 3, 7, 14, 21, 28, 56, 365]:
        df[f"lag_{L}"] = df[Y_COL].shift(L)

    # Rolling stats
    for W in [3, 7, 14, 28, 56]:
        df[f"roll_mean_{W}"] = df[Y_COL].shift(1).rolling(W).mean()
        df[f"roll_std_{W}"] = df[Y_COL].shift(1).rolling(W).std()

    # Exponentially weighted moving average
    df["ewm_mean_14"] = df[Y_COL].shift(1).ewm(span=14, adjust=False).mean()
    df["ewm_mean_28"] = df[Y_COL].shift(1).ewm(span=28, adjust=False).mean()

    # One day change
    df['volume_change_1d'] = df[Y_COL].shift(1) - df[Y_COL].shift(2)

    return df

def add_real_external_features(df: pd.DataFrame, weather_df: pd.DataFrame):
    """
    Merge real weather dataframe into the main dataframe
    """

    df = df.merge(weather_df, left_on=DATE_COL, right_on='date', how='left')
    df = df.drop('date', axis=1, errors='ignore')
    
    # Fill missing weather values with interpolation
    weather_cols = [c for c in df.columns if c.startswith('weather_')]
    for col in weather_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    return df



print("Loading all data...")
df_train = pd.read_csv('tsa_train.csv', parse_dates=[DATE_COL])
df_test = pd.read_csv('tsa_test.csv', parse_dates=[DATE_COL])

start_date = min(df_train[DATE_COL].min(), df_test[DATE_COL].min())
end_date = max(df_train[DATE_COL].max(), df_test[DATE_COL].max())

# Load weather data
weather_df = pd.read_csv('weather_data.csv', parse_dates=['date'])

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")
print(f"Weather data shape: {weather_df.shape}")

# Combine all data for consistent feature engineering
df_combined = pd.concat([df_train, df_test], ignore_index=True)
df_combined = df_combined.sort_values(DATE_COL).reset_index(drop=True)

# Add all features
print("Adding all features...")
df_combined = add_time_features(df_combined)
df_combined = add_holiday_week_features(df_combined)
df_combined = add_real_external_features(df_combined, weather_df)
df_combined = add_lag_roll_features(df_combined)
print(df_combined.head())

# Split back into train and test
print("\nSplitting back into train and test...")
df_train_processed = df_combined[df_combined[DATE_COL].isin(df_train[DATE_COL])].copy()
df_test_processed = df_combined[df_combined[DATE_COL].isin(df_test[DATE_COL])].copy()

print(f"Processed train shape: {df_train_processed.shape}")
print(f"Processed test shape: {df_test_processed.shape}")


print("\nPreparing training data...")

# Log transform target
df_train_processed["y_log"] = np.log1p(df_train_processed[Y_COL])

# Define features (exclude date, target, and log target)
feature_cols = [c for c in df_train_processed.columns if c not in [DATE_COL, Y_COL, "y_log"]]
print("Featured cols: ", feature_cols)

# Split up data into X and Y
X_train = df_train_processed[feature_cols]
y_train = df_train_processed["y_log"]

# Give weights to extreme volume days
peak_threshold = np.percentile(y_train, 95)
sample_weight = np.where(
    y_train >= peak_threshold,
    1.5,
    1.0
)

# Define Model
model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.5,
    objective="reg:squarederror",
)

#Train Model
print("\nTraining XGBoost model...")
model.fit(X_train, y_train, sample_weight=sample_weight)
print("Model training complete!")

# Test Model
print("\nMaking predictions...")
X_test = df_test_processed[feature_cols]
y_pred_log = model.predict(X_test)

# Convert back from log scale
y_pred = np.expm1(y_pred_log)

# Add predictions to test dataframe
df_test_processed['predicted_volume'] = y_pred

# Save to CSV
output_file = 'predictions.csv'
df_test_processed[[DATE_COL, 'predicted_volume']].to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'")


print("Creating actual vs predicted visualization...")
# Merge actual values with predictions
results = df_test_processed[[DATE_COL, 'predicted_volume']].copy()
results['actual_volume'] = df_test[Y_COL].values

# Calculate error metrics
mae = mean_absolute_error(results['actual_volume'], results['predicted_volume'])
rmse = np.sqrt(mean_squared_error(results['actual_volume'], results['predicted_volume']))
mape = np.mean(np.abs((results['actual_volume'] - results['predicted_volume']) / results['actual_volume'])) * 100

print(f"\nTest Set Performance:")
print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")

# Create the plot
plt.figure(figsize=(15, 7))

# Plot actual values
plt.plot(results[DATE_COL], results['actual_volume'], 
            label='Actual Volume', color='blue', linewidth=2, marker='o', markersize=4, alpha=0.7)

# Plot predicted values
plt.plot(results[DATE_COL], results['predicted_volume'], 
            label='Predicted Volume', color='red', linewidth=2, marker='s', markersize=4, alpha=0.7)

plt.xlabel('Date')
plt.ylabel('Volume')
plt.title(f'Test Set: Actual vs Predicted Volume\nMAE: {mae:,.0f} | RMSE: {rmse:,.0f} | MAPE: {mape:.2f}%', 
            fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save and show
plt.savefig('test_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to 'test_actual_vs_predicted.png'")

# Also create a scatter plot to see correlation
plt.figure(figsize=(8, 8))
plt.scatter(results['actual_volume'], results['predicted_volume'], alpha=0.5, s=50)

# Add diagonal line for perfect predictions
min_val = min(results['actual_volume'].min(), results['predicted_volume'].min())
max_val = max(results['actual_volume'].max(), results['predicted_volume'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Volume', fontsize=12)
plt.ylabel('Predicted Volume', fontsize=12)
plt.title('Actual vs Predicted: Scatter Plot', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('test_scatter_plot.png', dpi=300, bbox_inches='tight')
print("Scatter plot saved to 'test_scatter_plot.png'")
