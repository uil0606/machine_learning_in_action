from numpy import *
import matplotlib.pyplot as plt


def load_dataset():
    data_mat = []
    label_mat = []
    fr = open('C:/Users/magfi/Documents/My github/machinelearninginaction/machinelearninginaction/Ch05/testset.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(inX):
    return longfloat(1.0 / (1 + exp(-inX)))


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
    y = (-weights[0] - x * weights[1]) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def stoc_grad_ascent(data_mat, class_labels):
    m, n = shape(data_mat)
    alpha = 1.0
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_mat[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * array(data_mat[i])  # list与np.array
    return weights


def stoc_grad_ascent1(data_mat, class_labels, max_iter=150):
    m, n = shape(data_mat)
    alpha = 1.0
    weights = ones(n)
    for j in range(max_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + i + j) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[i] * weights))
            error = class_labels[i] - h
            weights = weights + alpha * error * array(data_mat[i])  # list与np.array
            del (data_index[rand_index])
    return weights


def classify_vec(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colic_test():
    fr_train = open(
        'C:/Users/magfi/Documents/My github/machinelearninginaction/machinelearninginaction/Ch05/horseColicTraining.txt')
    fr_test = open(
        'C:/Users/magfi/Documents/My github/machinelearninginaction/machinelearninginaction/Ch05/horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    training_weights = stoc_grad_ascent1(array(training_set), array(training_labels), 5)  # num of iters
    error_count = 0
    num_test_vec = 0
    for line in fr_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vec(array(line_arr), training_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = float(error_count) / num_test_vec
    print('the error rate is: %f' % error_rate)
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print('after %d iterations the average error rate is: %f' % (num_tests, error_sum / float(num_tests)))
    return


def main1():
    # data_arr, label_mat = load_dataset()
    # weights = stoc_grad_ascent1(data_arr, label_mat)
    # plot_best_fit(weights)
    # multi_test()
    colic_test()


if __name__ == '__main__':
    main1()
