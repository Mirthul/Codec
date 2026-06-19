import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("sales_data.csv", encoding="latin1")

print("Columns:")
print(df.columns)

# =====================================================
# CONVERT ORDERDATE TO DATETIME
# =====================================================

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

# =====================================================
# CREATE DAILY SALES DATA
# =====================================================

daily_sales = df.groupby('ORDERDATE')['SALES'].sum().reset_index()

daily_sales.columns = ['ds', 'y']

print(daily_sales.head())

# =====================================================
# VISUALIZE HISTORICAL SALES
# =====================================================

plt.figure(figsize=(12,6))
plt.plot(daily_sales['ds'], daily_sales['y'])
plt.title("Historical Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

split_index = int(len(daily_sales) * 0.8)

train = daily_sales[:split_index]
test = daily_sales[split_index:]

# =====================================================
# PROPHET MODEL
# =====================================================

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model.fit(train)

# =====================================================
# FORECAST TEST DATA
# =====================================================

future = model.make_future_dataframe(
    periods=len(test),
    freq='D'
)

forecast = model.predict(future)

# =====================================================
# PLOT FORECAST
# =====================================================

fig1 = model.plot(forecast)
plt.title("Sales Forecast")
plt.show()

# =====================================================
# TREND + SEASONALITY
# =====================================================

fig2 = model.plot_components(forecast)
plt.show()

# =====================================================
# FORECAST VS ACTUAL
# =====================================================

forecast_test = forecast[['ds','yhat']].tail(len(test))

actual = test['y'].values
predicted = forecast_test['yhat'].values

plt.figure(figsize=(14,6))

plt.plot(
    test['ds'],
    actual,
    label='Actual Sales'
)

plt.plot(
    test['ds'],
    predicted,
    label='Forecast Sales'
)

plt.title("Forecast vs Actual Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# EVALUATION
# =====================================================

mae = mean_absolute_error(actual, predicted)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

mape = np.mean(
    np.abs((actual - predicted) / actual)
) * 100

print("\n======================")
print("MODEL EVALUATION")
print("======================")
print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("MAPE:", round(mape,2), "%")

# =====================================================
# FUTURE FORECAST - NEXT 30 DAYS
# =====================================================

future_30 = model.make_future_dataframe(
    periods=30,
    freq='D'
)

future_forecast = model.predict(future_30)

print("\nNext 30 Days Forecast")

print(
    future_forecast[['ds','yhat']].tail(30)
)

# =====================================================
# FUTURE FORECAST GRAPH
# =====================================================

plt.figure(figsize=(12,6))

plt.plot(
    future_forecast['ds'],
    future_forecast['yhat']
)

plt.title("Future Sales Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.show()
