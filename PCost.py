import numpy as np

def load_data_set():
    data_arr = []
    label_arr = []
    f = open('../TestSet01.txt', 'r')
    for line in f.readlines():
        line_arr = line.strip().split()
        data_arr.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
        label_arr.append(int(line_arr[2]))
    return data_arr, label_arr

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def grad_ascent(data_arr, class_labels, c10, c01):
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_mat)
    alpha = 0.001
    max_cycles = 5000
    weights = np.ones((n, 1))
    one = np.mat(np.ones((m, 1)))
    for k in range(max_cycles):
        z = sigmoid(data_mat * weights)
        h = np.multiply(z, (one - z))
        error = np.multiply((one * c01 - (c01 + c10) * label_mat), h)
        weights = weights - alpha * data_mat.transpose() * error
    return weights

def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
    x = np.arange(.0, 4.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

def test():
    data_arr, class_labels = load_data_set()
    weights = grad_ascent(data_arr, class_labels, 1, 10).getA()
    print(weights)
    plot_best_fit(weights)

test()




# mat1 = np.mat([[1],
#        [3]])
# mat2 = np.mat([[2],
#               [4]])
# r = np.multiply(mat1, mat2)
# print(r)