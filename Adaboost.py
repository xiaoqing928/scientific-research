import numpy as np
from sklearn import linear_model

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
        data_arr.append([np.float(line_arr[0]), np.float(line_arr[1])])
        label_arr.append(int(line_arr[2]))
        count += 1
    return data_arr, label_arr

#LR模型
def model_lr(X, y, w):
    lr = linear_model.LogisticRegression(solver='liblinear')
    lr.fit(X, y, sample_weight=w)
    acc = lr.score(X, y)
    return acc, lr.predict(X)

def Adaboost(x, y):
    n = len(x)
    #迭代次数
    m = 10
    pre_list = []
    alpha_list = []
    # 初始化权重
    w = np.array([1/n] * n)
    print(w)
    for i in range(m):
        acc, prey = model_lr(x, y, w*n)
        err = 1 - acc
        print(err)
        if err > 0.5:
            break
        alpha = 1/2 * np.log((1-err)/err)
        for i in range(len(x)):
            if y[i] == prey[i]:
                w[i] = w[i] * np.exp(-alpha)
            else:
                w[i] = w[i] * np.exp(alpha)
        # w = w * np.exp(-alpha * np.array(y) * prey)
        z = np.sum(w)
        w = w / z
        print(w)
        alpha_list.append(alpha)
        pre_list.append(prey)
    result = []
    for i in range(len(pre_list[0])):
        sum = 0
        for j in range(len(pre_list)):
            sum += alpha_list[j] * pre_list[j][i]
        if sum < 0.5:
            result.append(0)
        else:
            result.append(1)
    return result

path = '../TestSet01.txt'
X, y = load_data_set(path)
r = Adaboost(X, y)
def cost(y_pre, y):
    n01, n10 = 0, 0
    for i in range(len(y)):
        if y[i] == 0 and y_pre[i] == 1:
            n01 += 1
        if y[i] == 1 and y_pre[i] == 0:
            n10 += 1
    print(n01, n10)
cost(r, y)
