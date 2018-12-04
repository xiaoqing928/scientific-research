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


def Adaboost(x, y, c01, c10):
    # 代价加进数据集
    x = np.array(x)
    cost = []
    for i in range(len(y)):
        if y[i] == 1:
            cost.append([c10])
        else:
            cost.append([c01])
    x = np.append(x, np.array(cost), axis=1)
    #迭代次数
    m = 10
    pre_list = []
    alpha_list = []
    # 初始化权重
    w_sum = np.sum(x[:, 2])
    w = x[:, 2] / w_sum
    print(w)
    for i in range(m):
        acc, prey = model_lr(x[:, :2], y, w*len(y))
        err = 1 - acc
        print('错误率{}'.format(err))
        alpha = 1/2 * np.log((1-err)/err)
        for i in range(len(x)):
            # 判对的乘alpha和对应的代价
            if y[i] == 1 and prey[i] == 1:
                w[i] = w[i] * np.exp(-alpha) * c10
            elif y[i] == 0 and prey[i] == 0:
                w[i] = w[i] * np.exp(-alpha) * c01
            # 判错的-alpha和对应的代价
            elif y[i] == 1 and prey[i] == 0:
                w[i] = w[i] * np.exp(alpha) * c10
            else:
                w[i] = w[i] * np.exp(alpha) * c01
        # w = w * np.exp(-alpha * np.array(y) * prey)
        z = np.sum(w)
        w = w / z
        print(w)
        alpha_list.append(alpha)
        pre_list.append(prey)
        print(prey)
    result = []
    for i in range(len(pre_list[0])):
        sum = 0
        for j in range(len(pre_list)):
            if pre_list[j][i] == 0:
                sum += alpha_list[j] * (pre_list[j][i]-1)
            else:
                sum += alpha_list[j] * pre_list[j][i]
        if sum < 0:
            result.append(0)
        else:
            result.append(1)
    return result

def cost(y_pre, y, c01, c10):
    n01, n10 = 0, 0
    for i in range(len(y)):
        if y[i] == 0 and y_pre[i] == 1:
            n01 += 1
        if y[i] == 1 and y_pre[i] == 0:
            n10 += 1
    print('0判成1的数量为：{0},代价为{1}'.format(n01, n01*c01))
    print('1判成0的数量为：{0},代价为{1}'.format(n10, n10*c10))
    print('总代价为：{}'.format(n01*c01 + n10*c10))


    return 0

path = '../TestSet01.txt'
X, y = load_data_set(path)
# X = np.array(X)
# cost = []
# for i in range(len(y)):
#     if y[i] == 1:
#         cost.append([1])
#     else:
#         cost.append([10])
# X = np.append(X, np.array(cost), axis=1)
# print(np.sum(X[:,2]))
# print(np.append(X, y, axis=0))

#稍微相差太大偏离太多了
# r = Adaboost(X, y, 1.5, 1)
# cost(r, y, 1.5, 1)
c01, c10 = 1, 10
r = Adaboost(X, y, c01, c10)
cost(r, y, c01, c10)

