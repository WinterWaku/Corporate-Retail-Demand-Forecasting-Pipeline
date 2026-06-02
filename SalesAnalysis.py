import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from prophet import Prophet

#       Initial Health Checks
# Load the data
df = pd.read_csv('sales.csv')

# Quick Overview of rows, columns, and data types
print("--- Data Overview ---")
df.info()

# Count exact number of missing values per column
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Check for duplicate rows
duplicate_count = df.duplicated().sum()
print(f"\nTotal duplicate rows: {duplicate_count}")



#       Time-Series Checks
# Convert date column to datetime formay
df['date'] = pd.to_datetime(df['date'])

# Check for duplicate dates
date_duplicates = df.duplicated(subset=['date']).sum()
print(f"\nDuplicate Dates Count: {date_duplicates}")

# Check for Chronological Order
is_sorted = df['date'].is_monotonic_increasing
print(f"\nIs the data chronologically sorted?: {is_sorted}")

if not is_sorted:
    df = df.sort_values('date').reset_index(drop=True)
    print("\nData has been sorted chronologically.")



#       Numerical Integrity and Outliers
# Statistical summary of numeric columns
print("\n--- Numerical Summary ---")
print(df.describe())

# Flag negavtive or zero sales values
negative_sales = df[df['weekly_sales'] <= 0]
print(f"\nNegative or Zero Sales Count: {len(negative_sales)}")

# Outlier Detection using IQR
Q1 = df['weekly_sales'].quantile(0.25)
Q3 = df['weekly_sales'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['weekly_sales'] < lower_bound) | (df['weekly_sales'] > upper_bound)]
print(f"\nStatistical Outliers Dectected: {len(outliers)}")



# Load the data
df = pd.read_csv('sales.csv')
df['date'] = pd.to_datetime(df['date'])

# Aggregate to total company weekly sales
company_weekly_sales = df.groupby('date')['weekly_sales'].sum().reset_index()

# Check the trend
plt.figure(figsize=(12, 5))
plt.plot(company_weekly_sales['date'], company_weekly_sales['weekly_sales'], marker='o', linestyle='-')
plt.title('Total Company Weekly Sales Over Time Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# Check the seasonality
company_weekly_sales['month'] = company_weekly_sales['date'].dt.month
monthly_avg = company_weekly_sales.groupby('month')['weekly_sales'].mean().reset_index()

plt.figure(figsize=(8, 4))
plt.bar(monthly_avg['month'], monthly_avg['weekly_sales'], color='skyblue')
plt.title('Monthly Average Sales (Seasonality)')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()

# Handle anomalies / zero sales
min_sales = company_weekly_sales['weekly_sales'].min()
print(f'Minimum weekly sales: {min_sales}')

# Check if any week dropped drastically below normal baseline
anomaly_threshold = company_weekly_sales['weekly_sales'].median() * 0.3
anomalies = company_weekly_sales[company_weekly_sales['weekly_sales'] < anomaly_threshold]
print(f'Number of weeks with sales below {anomaly_threshold}: {len(anomalies)}')

# Check for missing weeks
missing_weeks = pd.date_range(start=company_weekly_sales['date'].min(), end=company_weekly_sales['date'].max())
missing_weeks = missing_weeks[~missing_weeks.isin(company_weekly_sales['date'])]
print(f'Number of missing weeks: {len(missing_weeks)}')

# Check for outliers
outliers = company_weekly_sales[company_weekly_sales['weekly_sales'] > 1000000]
print(f'Number of weeks with sales above $1,000,000: {len(outliers)}')

# Format for Prophet
# Prophet requires exactly two columns: 'ds' (datestamp) and 'y' (target value)
prophet_df = company_weekly_sales[['date', 'weekly_sales']].rename(columns={'date': 'ds', 'weekly_sales': 'y'})

# Ensure data is sorted chronologically
prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

# data splitting (The Golden Rule)
# Check the date range available
min_date = prophet_df['ds'].min()
max_date = prophet_df['ds'].max()
print(f"Dataset ranges from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

# Your data spans roughly 3 full years (2022, 2023, 2024)
# Let's use the final 3 months (September 2024 to December 2024) as the holdout test set
split_date = '2024-09-01'

train_data = prophet_df[prophet_df['ds'] < split_date]
test_data = prophet_df[prophet_df['ds'] >= split_date]

print(f"Training set rows: {len(train_data)}")
print(f"Testing set rows (Holdout Set): {len(test_data)}")


#       Train, Forecast, and Evaluate
# Set logging control
logging.getLogger('prophet').setLevel(logging.ERROR)

# Initialize the Prophet model
model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True,
    interval_width=0.95
)

# Fit the model on the training data
model.fit(train_data)

# Make predictions on the test data
forecast = model.predict(test_data[['ds']])

# Merge the predictions back with the actual values to compare them easily
evaluation_df = pd.merge(test_data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_data['ds'], train_data['y'], label='Historical Training Data', color='gray', alpha=0.6)
plt.plot(evaluation_df['ds'], evaluation_df['y'], label='Actual Sales (Holdout Set)', color='blue', marker='o')
plt.plot(evaluation_df['ds'], evaluation_df['yhat'], label='AI Predicted Sales', color='red', linestyle='--', marker='x')

# Add shaded uncertainty interval (excellent representation for stock buffering)
plt.fill_between(evaluation_df['ds'], evaluation_df['yhat_lower'], evaluation_df['yhat_upper'], color='red', alpha=0.15, label='95% Confidence Buffer')

plt.title('AI Demand Forecast vs Actual Sales Performance')
plt.xlabel('Date')
plt.ylabel('Total Company Sales')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate evaluation metrics
def calculate_metrics(df_eval):
    actual = df_eval['y']
    predicted = df_eval['yhat']
    
    # MAPE: Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    return mape, rmse

mape, rmse = calculate_metrics(evaluation_df)
print("\n=== MODEL PERFORMANCE METRICS ===")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Forecast Accuracy Rate: {100 - mape:.2f}%")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")