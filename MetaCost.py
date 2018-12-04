import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt

'''
一、数据采样
'''
# 加载数据集
def load_data_set(path):
    '''
    :param path:文件路径
    :return: 数据集和标签
    '''
    data_arr = []
    label_arr = []
    f = open(path)
    count = 1 #设定从1开始，给每个样本一个标记
    #读行
    for line in f.readlines():
        line_arr = line.strip().split()
        data_arr.append([np.float(line_arr[0]), np.float(line_arr[1]), count])
        label_arr.append(int(line_arr[2]))
        count += 1
    return data_arr, label_arr

# 切数据
def splitData(X, y, seed):
    '''
    :param X: 数据集
    :param y: 对应的label
    :param seed: 随机种子（每次给定不同的seed）
    :return: 切好的数据集和对应的label
    '''
    # 每次取样取（1-0.4）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
    return X_train, y_train

'''
二、模型
'''
# LR模型
def model_lr(X, y):
    '''
    :param X: 数据集（最后一列记录对应的行）
    :param y: 标签
    :return: key=行数，value=[概率，1]的列表，后面统计次数
    '''
    lr = linear_model.LogisticRegression(solver='liblinear')
    #转成矩阵形式，方便切片
    X_mat = np.mat(X)
    lr.fit(X_mat[:,:2], y)
    p_mat = lr.predict_proba(X_mat[:,:2])#得到每个点取0、1的概率
    #构建 第几个样本-对应概率之间的关系
    dictp = {}
    for i in range(len(X_mat)):
        index = int(X_mat.getA()[i][2]) #矩阵转数组，取索引
        p = p_mat[:,1][i]
        dictp[index] = np.array([p, 1])
        # print(int(X_mat.getA()[0][2]))
    # return p_mat[:,1]
    return dictp, lr.coef_, lr.intercept_

'''
三、bagging的思想
'''
def bagging(X, y, n):
    # 累加记录所有的样本和其平均概率
    dictall = {}
    for seed in range(n):
        #取样
        trainX, trainy = splitData(X, y, seed*10)
        dictr, w, b = model_lr(trainX, trainy)
        #求和
        for i in dictr.keys():
            if i not in dictall:
                dictall[i] = dictr[i]
            else:
                dictall[i] += dictr[i]
    for i in dictall.keys():
        dictall[i] = dictall[i][0]/dictall[i][1]
    for i in range(1, 2001):
        if i not in dictall.keys():
            dictall[i] = 1 / len(trainX)
    return dictall

# 重标记
def remark(dict, c01, c10):
    '''
    :param dict: key=第几个样本，value=该样本综合概率
    :param c01:0判1的代价
    :param c10: 1判0的代价
    :return: 重标记后的label
    '''
    for i in dict.keys():
        p = dict[i]
        #满足贝叶斯最小风险的条件
        if p * c10 >= (1 - p) * c01:
            dict[i] = 1
        else:
            dict[i] = 0
    return dict

# 画图
def plot_best_fit(weights, b):
    '''
    :param 分界线的参数:
    :return:
    '''
    data_mat, label_mat = load_data_set(path)
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 0])
            y_cord1.append(data_arr[i, 1])
        else:
            x_cord2.append(data_arr[i, 0])
            y_cord2.append(data_arr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
    x = np.arange(-1, 5.0, 0.1)
    y = (-b - weights[0][0] * x) / weights[0][1]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.grid()
    plt.show()

#测试函数
def test(path, c01, c10, n):
    X, y = load_data_set(path)
    # train, label = splitData(X, y, 10)
    # r = model_lr(train, label)
    r = bagging(X, y, n)
    # print(r)
    framea = pd.Series(remark(r, c01, c10))
    label = list(framea)
    #使用重标记的数据再次训练
    dict, w, b = model_lr(X, label)
    # print(w, b)
    plot_best_fit(w, b)
    #训练得到的结果
    n01, n10 = 0, 0
    for i in dict.keys():
        #0判成1
        a = dict[i][0]
        b = y[i-1]
        if dict[i][0] >= 0.5 and y[i-1] == 0:
            n01 += 1
        if dict[i][0] < 0.5 and y[i-1] == 1:
            n10 += 1
    print('0判成1的个数{0},代价是{1}'.format(n01,n01*c01))
    print('1判成0的个数{0},代价是{1}'.format(n10,n10*c10))
    print('总代价是{}'.format(n01*c01 + n10*c10))

path = '../TestSet01.txt'
c01, c10 = 10, 1
test(path, c01, c10, 10)

