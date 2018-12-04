import numpy as np

def load_data_set():
    data_arr = []
    label_arr = []
    f = open('../TestSet01.txt', 'r')
    for line in f.readlines():
        line_arr = line.strip().split()
        # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
        data_arr.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
        label_arr.append(int(line_arr[2]))
    return data_arr, label_arr

def sigmoid(data, weights):
    m = len(data)
    x = 0
    for i in range(m):
        x += data[i] * weights[i][0]
    return 1.0 / (1 + np.exp(-x))

def grad_ascentvector(data_arr, class_labels):
    m, n = np.shape(data_arr)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        for j in range(n):
             sum = 0 #求导之后的和
             #m行数据，遍历一遍
             for i in range(m):
                 h = sigmoid(data_arr[i], weights)
                 sum += (class_labels[i] - h) * data_arr[i][j]
             weights[j][0] = weights[j][0] + alpha * sum
    return weights


data, label = load_data_set()
weight = grad_ascentvector(data, label)
print(weight)