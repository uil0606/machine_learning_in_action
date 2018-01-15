from numpy import *


def load_dataset(file_name):
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_simple(data_mat_in, class_labels, C, toler, max_iter):
    data_mat = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    # print(label_mat.shape)
    b = 0
    m, n = shape(data_mat)
    alphas = zeros((m, 1))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            fXi = float(multiply(alphas, label_mat).T * (data_mat * data_mat[i, :].T)) + b
            Ei = fXi - float(label_mat[i])
            if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * Ei > toler) and alphas[i] > 0):
                j = select_j_rand(i, m)
                fXj = float(multiply(alphas, label_mat).T * (data_mat * data_mat[j, :].T)) + b
                Ej = fXj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(C, alphas[j] + alphas[i] - C)
                    H = min(0, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * data_mat[i, :] * data_mat[j, :].T - data_mat[i, :] * data_mat[i, :].T - data_mat[j,
                                                                                                    :] * data_mat[j,
                                                                                                         :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                # print(alphas[j],alphas[j].shape)
                # print(label_mat[j],label_mat[j].shape,label_mat[j].reshape(1,))
                alphas[j] = alphas[j]-(Ei - Ej) / eta * array(label_mat[j]).reshape(1,)
                alphas[i] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] =alphas[i] + label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[i, :] * data_mat[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[j, :] * data_mat[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("iter: %d i: %d,pairs changed %d" % (iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


data_mat,label_mat=load_dataset('c:/users/magfi/desktop/testSet.txt')
b,alphas=smo_simple(data_mat,label_mat,0.6,0.001,40)
print(b,alphas)