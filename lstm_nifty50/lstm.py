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
print(data.head(2))
df=data[["close"]].values
print(df.shape)
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
print(y[:5])
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
model = Sequential()
model.add(LSTM(units=128, input_shape=(x_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100)
from sklearn.metrics import mean_squared_error
y_tr=model.predict(x_train)
y_tr=y_tr.reshape(-1,1)
xcharizard1=sc.inverse_transform(y_tr)
y_te=model.predict(x_test)
y_tr=y_te.reshape(-1,1)
xcharizard2=sc.inverse_transform(y_te)
window_size = 10
split = int(len(y) * 0.8)


import streamlit as st
st.title("NIFTY 50 LSTM Prediction")
st.subheader("Dataset Preview")
st.dataframe(data.head())
st.subheader("Dataset Shape")
st.write(data.shape)
train_mse = mean_squared_error(y_train, y_tr)
test_mse = mean_squared_error(y_test, y_te)
st.subheader("Model Performance")
st.write(f"Train MSE: {train_mse}")
st.write(f"Test MSE: {test_mse}")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data.index, data["close"], color="red", label="Original")
ax.plot(
    data.index[window_size:window_size+split],
    xcharizard1,
    color="blue",
    label="Train Prediction"
)
ax.plot(
    data.index[window_size+split:window_size+split+len(xcharizard2)],
    xcharizard2,
    color="green",
    label="Test Prediction"
)
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
ax.set_title("Train & Test Predictions")
st.subheader("Prediction Graph")
st.pyplot(fig)
