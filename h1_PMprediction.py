import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

'''
数据预处理
共计240天的数据
每天观察24小时
由于要用连续的九个小时数据预测接下来的一小时数据
故每天可提取15组数据
共计240X15=3600组数据
每组数据包含一个18x9的数组和一个输出值
'''

file_way = r'E:\learning source\ml_lihongyi\dataset\hw1\train.csv'
train_data = pd.read_csv(file_way, encoding='ANSI')
column = train_data['observation'].unique()
train_data = train_data.replace('NR', 0)
train_data = train_data.drop(['Date', 'stations', 'observation'], axis=1)
# print(train_data)
FeatureNum = 18
train_X = []
train_Y = []
for i in range(int(len(train_data) / FeatureNum)):
    x_i = train_data.iloc[i * FeatureNum:(i + 1) * FeatureNum, :]
    for j in range(15):
        x_ij = x_i.iloc[:, j:(j + 9)]
        y_ij = x_i.iloc[9, j + 1]
        train_X.append(np.array(x_ij, dtype='float'))
        train_Y.append(float(y_ij))

# print(list(train_X[0][0])+list(train_X[0][1]))
# print(train_Y)

# 绘制各个特征与PM2.5的散点图
'''
for j in range(18):
    x=[]
    y=[]
    for i in range(len(train_X)):
        x.append(train_X[i][j])
        y.append(train_X[i][9])
    plt.scatter(x,y,s=10)
    plt.title(column[j])
    plt.show()
'''

# 选取PM10,PM2.5,SO2作为模型特征，对应的索引为8，9，12

# 初始化参数
w = np.array([0.1] * 27)
b = 0.1
lr = 0.0001
lamda = 0.0001
# 梯度下降更新w,b
for i in random.sample(range(len(train_X)), 3000):
    X = np.array(list(train_X[i][8]) + list(train_X[i][9]) + list(train_X[i][12]))
    w_temp = w
    for j in range(len(w_temp)):
        w[j] -= lr * 2 * ((b + np.dot(w_temp, X.T)) - train_Y[i]) * w[j] + lamda * 2 * w[j]
    b -= lr * 2 * ((b + np.dot(w_temp, X.T)) - train_Y[i]) + 2 * lamda * b

# 求损失函数loss
sum = 0
for i in range(len(train_X)):
    X = np.array(list(train_X[i][8]) + list(train_X[i][9]) + list(train_X[i][12]))
    sum += ((b + np.dot(w, X.T)) - train_Y[i]) ** 2
loss = sum / len(train_X)
print(loss)

# 测试集预测
test_file = r'E:\learning source\ml_lihongyi\dataset\hw1\test.csv'
test_data = pd.read_csv(test_file, header=None, encoding='ANSI')
test_data = test_data.drop([0, 1], axis=1)
test_data = test_data.replace('NR', 0)
test_X = np.array(test_data, dtype='float')
Y_prediction = []
for i in range(int(len(test_X) / FeatureNum)):
    X_i = np.array(list(test_X[i * 18 + 8]) + list(test_X[i * 18 + 9]) + list(test_X[i * 18 + 12]))
    Y_i = np.dot(w, X_i) + b
    Y_prediction.append(Y_i)

# 输出数据
sub_file = r'E:\learning source\ml_lihongyi\dataset\hw1\sample_submission.csv'
sub_data = pd.read_csv(sub_file, encoding='ANSI')
sub_data['value'] = Y_prediction
output_path = r'C:\Users\12723\Desktop\submission.csv'
sub_data.to_csv(output_path, index=None)
