from numpy import *
import matplotlib.pyplot as plt


def load_dataset():
    data_mat = []
    label_mat = []
    fr = open('D:/mygithub/machinelearninginaction/machinelearninginaction/Ch05/testset.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def grad_ascent(data_mat_in, class_labels):
    data_mat = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    m, n = shape(data_mat)
    alpha = 0.001
    max_iter = 500
    weights = ones((n, 1))
    for i in range(max_iter):
        h = sigmoid(data_mat * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_mat.transpose() * error
    return weights


def plot_best_fit(weights):
    data_mat, label_mat = load_dataset()
    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    weights = array(weights)
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # y=(1-x)/2
    y = (-weights[0] - x * weights[1]) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


data_arr, label_mat = load_dataset()
weights = grad_ascent(data_arr, label_mat)
plot_best_fit(weights)
