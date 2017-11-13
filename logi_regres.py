from numpy import *


def load_dataset():
    data_mat = []
    label_mat = []
    fr = open('testset.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line[1])])
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
