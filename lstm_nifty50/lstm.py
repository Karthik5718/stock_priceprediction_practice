import pandas as pd
import numpy as np
import tensorflow as tf
import os
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "NIFTY 50_day.csv")
data = pd.read_csv(file_path)
data["date"] = pd.to_datetime(data["date"])
data.set_index("date", inplace=True)
df = data[["close"]].values
sc = StandardScaler()
df_scaled = sc.fit_transform(df)
window_size = 10
x = []
y = []
for i in range(window_size, len(df_scaled)):
    x.append(df_scaled[i-window_size:i, 0])
    y.append(df_scaled[i, 0])
x = np.array(x)
y = np.array(y)
split = int(len(x) * 0.8)
x_train = x[:split]
x_test  = x[split:]
y_train = y[:split]
y_test  = y[split:]
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
@st.cache_resource
def train_and_predict(x_train, y_train, x_test):
    model = Sequential([
        LSTM(128, input_shape=(x_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=100, verbose=0)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    return model, y_train_pred, y_test_pred
model, y_train_pred, y_test_pred = train_and_predict(
    x_train, y_train, x_test
)
y_train_inv = sc.inverse_transform(y_train.reshape(-1,1))
y_test_inv = sc.inverse_transform(y_test.reshape(-1,1))
y_train_pred_inv = sc.inverse_transform(y_train_pred)
y_test_pred_inv = sc.inverse_transform(y_test_pred)
train_mse = mean_squared_error(y_train_inv, y_train_pred_inv)
test_mse = mean_squared_error(y_test_inv, y_test_pred_inv)
train_mae = mean_absolute_error(y_train_inv, y_train_pred_inv)
test_mae = mean_absolute_error(y_test_inv, y_test_pred_inv)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train_inv, y_train_pred_inv)
test_r2 = r2_score(y_test_inv, y_test_pred_inv)
train_mape = np.mean(np.abs((y_train_inv - y_train_pred_inv) / y_train_inv)) * 100
test_mape = np.mean(np.abs((y_test_inv - y_test_pred_inv) / y_test_inv)) * 100
st.title("NIFTY 50 LSTM Prediction")
st.subheader("Dataset Preview")
st.dataframe(data.head())
st.subheader("Dataset Shape")
st.write(data.shape)
st.subheader("Train Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("MSE", f"{train_mse:.4f}")
col2.metric("RMSE", f"{train_rmse:.4f}")
col3.metric("MAE", f"{train_mae:.4f}")
col4, col5 = st.columns(2)
col4.metric("R2 Score", f"{train_r2:.4f}")
col5.metric("MAPE (%)", f"{train_mape:.2f}")
st.subheader("Test Metrics")
col6, col7, col8 = st.columns(3)
col6.metric("MSE", f"{test_mse:.4f}")
col7.metric("RMSE", f"{test_rmse:.4f}")
col8.metric("MAE", f"{test_mae:.4f}")
col9, col10 = st.columns(2)
col9.metric("R2 Score", f"{test_r2:.4f}")
col10.metric("MAPE (%)", f"{test_mape:.2f}")
split = int(len(y) * 0.8)
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data.index, data["close"], color="red", label="Original")
ax.plot(
    data.index[window_size:window_size+split],
    y_train_pred_inv,
    color="blue",
    label="Train Prediction"
)
ax.plot(
    data.index[window_size+split:window_size+split+len(y_test_pred_inv)],
    y_test_pred_inv,
    color="green",
    label="Test Prediction"
)
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
ax.set_title("Train & Test Predictions")
st.subheader("Prediction Graph")
st.pyplot(fig)
