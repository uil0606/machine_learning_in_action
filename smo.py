def load_dataset(file_name):
    data_mat=[]
    label_mat=[]
    fr= open(file_name)
    for line in fr.readlines():
        linr_arr=line.strip().split('\t')
        data_mat.apped([float(line_arr[0]),float(linr_arr[1])])
        label_mat.append(float(linr_arr[3]))
    return data_mat,label_mat


def select_j_rand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j


def clip_alpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj


def smo_simple(data_mat_in,class_labels,C,toler,max_iter):
    # from numpy import *
    data_mat=mat(data_mat_in)
    label_mat=mat(class_labels).transpose()
    b=0
    m,n=shape(data_mat)
    alphas=mat(zeros(m,1))
    iter=0
    while (iter<max_iter):
        alpha_pairs_changed=0
        for i in range(m):
            fXi=float(multiply(alphas,label_mat).T*(data_mat*data_mat[i,:].T))+b
            Ei=fXi-float(label_mat[i])
            if ((label_mat[i]*Ei< -toler) and (alphas[i]<C)) or ((label_mat[i]*Ei>toler) and alphas[i]>0):