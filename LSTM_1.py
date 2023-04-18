import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from keras import Sequential, layers, utils
import warnings
warnings.filterwarnings('ignore')
#讀取數據
dataset = pd.read_csv('raw_1.csv')
dataset.index = dataset.Time
dataset.drop(columns = ['Time'], axis = 0, inplace = True)

#數值歸一化
scaler = MinMaxScaler()
dataset['Value'] = scaler.fit_transform(dataset['Value'].values.reshape(-1,1))
print(dataset.head)
dataset['Value'].plot(figsize=(16,8))
plt.show()

#功能函數
def create_new_dataset(dataset, seq_len = 12):
    x = []
    y = []
    start = 0 #初始位置
    end = dataset.shape[0] - seq_len #截止位置
    for i in range(start, end):
        sample = dataset[i : i+seq_len] #基於時間跨度seq_len建立樣本
        label = dataset[i+seq_len] #創建sample對應的標籤
        x.append(sample)
        y.append(label)
    return np.array(x), np.array(y)

#功能函數:基於新的特徵的數據集和標籤集，切分訓練及測試集
def split_dataset(x, y ,train_ratio = 0.67):
    x_len = len(x) #特徵值採集樣本x的樣本數量
    train_data_len = int(x_len * train_ratio) #訓練集樣本數量

    x_train = x[:train_data_len] #訓練集
    y_train = y[:train_data_len] #訓練標籤集

    x_test = x[train_data_len:]
    y_test = y[train_data_len:]
    return x_train, x_test, y_train, y_test

#功能函數:創建批數據
def create_batch_data(x,y,batch_size = 32, data_type = 1):
    if data_type ==1: #測試集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y))) #封裝x,y ,成為tensor類型
        test_batch_data = dataset.batch(batch_size) #建構批數據
        return test_batch_data
    else: #訓練集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
        train_batch_data = dataset.cache().shuffle(1000).batch(batch_size)
        return train_batch_data

#原始數據集
dataset_original = dataset
print(" 原始數據集:", dataset_original.shape)
#構造特徵數據集和標籤集
SEQ_LEN = 20 #序列長度
x, y = create_new_dataset(dataset_original.values, seq_len = SEQ_LEN)
print(x.shape)
print(y.shape)
#樣本1 - 特徵數據
print(x[0])
print(y[0])
#數據切分
x_train , x_test, y_train, y_test = split_dataset(x, y, train_ratio=0.2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

batch_size = 64

#基於新的x_train, y_train, x_test, y_test創建批數據
test_batch_dataset = create_batch_data(x_test, y_test, batch_size, data_type=1)
train_batch_dataset = create_batch_data(x_test, y_test, batch_size, data_type=2)

#創建模型
model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape = (SEQ_LEN, 1)))
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.3))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#顯示模型結構
utils.plot_model(model)

#定義checkpiont
file_path = "best_checkpiont.hdfs"
checkpiont_callback = tf.keras.callbacks.ModelCheckpoint(filepath = file_path, monitor = 'loss',mode = 'min', save_best_only = True, save_weights_only = True )

#模型編譯
model.compile(optimizer='adam', loss = 'mae')

#模型訓練
history = model.fit(train_batch_dataset, epochs=10, validation_data = test_batch_dataset, callbacks=[checkpiont_callback])

#顯示train loss 和 val loss
plt.figure(figsize = (16,8))
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val loss')
plt.title("LOSS")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc = 'best')
plt.show()

#模型驗證
test_pred = model.predict(x_test, verbose=1)
print(test_pred.shape)
print(y_test.shape)
#計算R2
import math
score = r2_score(y_test, test_pred)
print("r2:", score)
RMSE = math.sqrt(mean_squared_error(y_test, test_pred))
print("RMSE: " , RMSE)
MAE = mean_absolute_error(y_test, test_pred)
print("MAE: " , MAE)
MAPE = mean_absolute_percentage_error(y_test, test_pred)
print("MAPE: " , MAPE)

#繪製模型驗證結果
plt.figure(figsize=(16,8))
plt.plot(y_test, label = "True label")
plt.plot(test_pred, marker = '*', label = "Pred label")
plt.title("True vs Pred")
plt.legend(loc = 'best')
plt.show()

# from sklearn.linear_model import LinearRegression
# #R2圖
# fig, ax = plt.subplots()
# ax.scatter(y_test[0],test_pred[:,0])
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Actual')
# ax.set_ylabel('Predicted')
# #regression line
# y_test, y_predicted = y_test.reshape(-1,1), test_pred.reshape(-1,1)
# ax.plot(y_test, LinearRegression().fit(y_test, y_predicted).predict(y_test))

#繪製test中前10000個點的真值與預測值
y_true = y_test[:30000]
y_pred = test_pred[:30000]
fig, axes = plt.subplots(2,1, figsize = (16,8))
axes[0].plot(y_true, marker = 'o', color = 'red')
axes[1].plot(y_pred, marker = '*', color = 'blue')
plt.show()

# #選擇test中最後一個樣本
# print(x_test.shape)
# sample = x_test[0]
# print(sample.shape)
# sample = sample.reshape(1, sample.shape[0], 1)
# print(sample.shape)

# #模型預測
# sample_pred = model.predict(sample)
# sample_pred

# #預測後續20秒的數據
# ture_data = x_test[-1]
# print(ture_data)

# print(len(ture_data.shape))
# print(list(ture_data[:,0]))

# def predict_next(model, sample, epochs=20):
#     templ = list(sample[:,0])
#     for i in range(epochs):
#         sample = sample.reshape(1,SEQ_LEN, 1)
#         pred = model.predict(sample)
#         value = pred.tolist()[0][0]
#         templ.append(value)
#         sample = np.array(templ[i+1:i+SEQ_LEN+1])
#     return templ

# preds = predict_next(model, ture_data, 1000)

# plt.figure(figsize=(12,6))
# plt.plot(preds,color = 'red', label = 'Prediction')
# plt.plot(ture_data, color = 'blue', label = 'Truth')
# plt.xlabel("Epochs")
# plt.ylabel("value")
# plt.legend(loc = 'best')
# plt.show()



