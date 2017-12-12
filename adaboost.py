import matplotlib.pyplot as plt
from numpy import *


def load_simple_data():
    data_mat = matrix([[1, 2.1], [2, 1.1], [1.3, 1], [1, 1], [2, 1]])
    class_labels = [1, 1, -1, -1, 1]
    return data_mat, class_labels



def main():
    data_mat, class_labels = load_simple_data()
    fig = plt.figure(figsize=(8, 8), dpi=88)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    for i in range(len(class_labels)):
        if class_labels[i] == 1:
            type1_x.append(data_mat[i, 0])
            type1_y.append(data_mat[i, 1])
        if class_labels[i] == -1:
            type2_x.append(data_mat[i, 0])
            type2_y.append(data_mat[i, 1])

    plt.scatter(type1_x, type1_y,s=100, c='red',marker='o')
    plt.scatter(type2_x, type2_y,s=100, c='green',marker='x')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()

