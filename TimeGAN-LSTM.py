# -*- coding: utf-8 -*-
"""
TimeGAN-LSTM framework for Dissolved Prediction.
Last updated Date: November 24th 2024
Code author: Gang Li: gangli_rcee@163.com

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False


# Read generated data
data = np.load('generated_data.npy')
print('Dimension of ori_data: ', data.shape)

# Converts data to a two-dimensional array
data1 = data.reshape(-1, 10)
print('Dimension of the array: ', data1.shape)

#########The length of the generated data used for the LSTM training#########
data_extract = data1[0:int(6576),:]
print('generated_data dimension added to the training set: ',data_extract.shape)
print('generated_data length added to the training set：',len(data_extract))


# Read features of generated_data
x1_feature = data_extract[:, 1:10].reshape(-1,1,9)

# Read lables of generated_data
x1_label = data_extract[:, 0:1]

df = pd.read_csv('ori_data.csv')
# Read features of ori_data
X_features = df.iloc[:, 2:11]
print('Feature dimensions of original data',X_features.shape)
scalerX = MinMaxScaler().fit(X_features)
X_features =scalerX.transform(X_features)

# Read lables of ori_data
Y_labels = np.array(df.iloc[:,1]).reshape(-1, 1)
scalerY = MinMaxScaler().fit(Y_labels)
Y_labels = scalerY.transform(Y_labels)


inv_X1_feature = scalerX.inverse_transform(data1[:,0:9])
inv_X1_lable = scalerY.inverse_transform(data1[:,-1].reshape(-1,1))

# Division of training dataset and testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_features, Y_labels, test_size=0.3, random_state=42, shuffle=False)


# 将数据转换为 LSTM 所需的输入形状
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

x_train=np.concatenate((X_train, x1_feature), axis=0)
y_train=np.concatenate((y_train, x1_label), axis=0)

print('feature dimension of training set: ', x_train.shape)
print('label dimension of training set: ', y_train.shape)
print('feature dimension of testing set: ', X_test.shape)
print('label dimension of testing set: ', y_test.shape)

# LSTM-model
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(1, x_train.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Model training
history = model.fit(x_train, y_train, epochs=100, batch_size=24, validation_data = (X_test, y_test), verbose=1)

# Training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# Predictions during the training period
ytrain_pred = model.predict(x_train)
inv_ytrain_pred = scalerY.inverse_transform(ytrain_pred)
inv_ytrain = scalerY.inverse_transform(y_train)

# Predictions during the testing period
ytest_pred = model.predict(X_test)#.reshape(-1,)
inv_ytest_pred = scalerY.inverse_transform(ytest_pred)
inv_ytest = scalerY.inverse_transform(y_test)

print(inv_ytrain.shape,inv_ytrain_pred.shape,inv_ytest.shape,inv_ytest_pred.shape)
# Save the observations and predictions
pd.DataFrame(np.concatenate((inv_ytrain, inv_ytrain_pred), axis=1)).to_csv('0T_train.csv')
pd.DataFrame(np.concatenate((inv_ytest, inv_ytest_pred), axis=1)).to_csv('0T_test.csv')


# Visualization of the predictive process and observed process
time_column = df.iloc[0:len(inv_ytest_pred), 0]
plt.plot(time_column, inv_ytest, label='observations')
plt.plot(time_column, inv_ytest_pred, label='predictions')
plt.legend()
plt.xlabel('Time')
plt.ylabel('DO')
plt.title('TimeGAN-LSTM-Results')
plt.show()

# METRICS CALCULATION
rmse = np.sqrt(mean_squared_error(inv_ytest, inv_ytest_pred))
print("RMSE:", rmse)

mae = mean_absolute_error(inv_ytest, inv_ytest_pred)
print("MAE:", mae)

mse = mean_squared_error(inv_ytest, inv_ytest_pred)
print("MSE:", mse)

R2 = r2_score(inv_ytest, inv_ytest_pred)
print("R2:", R2)
