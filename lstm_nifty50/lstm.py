import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pandas as pd
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "NIFTY 50_day.csv")
data = pd.read_csv(file_path)
data["date"]=pd.to_datetime(data["date"])
data.set_index("date", inplace=True)
df=data[["close"]].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df=sc.fit_transform(df)
window_size=10
x=[]
y=[]
for i in range(window_size,len(df)):
    x.append(df[i-window_size:i,0])
    y.append(df[i,0])
x=np.array(x)
y=np.array(y)
split = int(len(x) * 0.8)
x_train = x[:split]
x_test  = x[split:]
y_train = y[:split]
y_test  = y[split:]
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras import layers, regularizers
import streamlit as st
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
@st.cache_resource
def train_and_predict(x_train, y_train, x_test, sc):
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100, verbose=0)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    y_train_pred_inv = sc.inverse_transform(y_train_pred)
    y_test_pred_inv = sc.inverse_transform(y_test_pred)
    return y_train_pred_inv, y_test_pred_inv, train_mse, test_mse
y_train_pred_inv, y_test_pred_inv, train_mse, test_mse = train_and_predict(
    x_train, y_train, x_test, sc
)
st.title("NIFTY 50 LSTM Prediction")
st.subheader("Dataset Preview")
st.dataframe(data.head())
st.subheader("Dataset Shape")
st.write(data.shape)
st.subheader("Model Performance")
st.write(f"Train MSE: {train_mse}")
st.write(f"Test MSE: {test_mse}")
window_size = 10
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
