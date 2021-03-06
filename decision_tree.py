from math import log
import operator
import copy


def calc_shannon_ent(dataset):
    #  dataset最后一列为标签，每个样本等长，某些特征可以为空
    #   根据y，计算集合的熵
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def calc_gini(dataset):
    #   根据y,计算集合的gini值
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    gini = 1
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        gini -= prob ** 2
    return gini


def split_dataset(dataset, axis, value):
    #  去掉第axis列
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def choose_split_feature(dataset):
    #  只针对离散数据集可用，连续变量不可用
    #  dataset最后一列为标签，每个样本等长，某些特征可以为空，空值也作为一个值
    #  一个特征只做一次分支节点
    num_feat = len(dataset[0]) - 1
    base_ent = calc_shannon_ent(dataset)
    best_infogain = 0
    best_feat = -1
    for i in range(num_feat):
        feat_list = [a[i] for a in dataset]
        uniq_val = set(feat_list)  # set是获得列表中唯一值的最快方法
        new_ent = 0
        for value in uniq_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))  # float
            new_ent += prob * calc_shannon_ent(sub_dataset)
        infogain = base_ent - new_ent
        if infogain > best_infogain:
            best_infogain = infogain
            best_feat = i
    return best_feat


def major_cnt(class_list):
    class_cnt = {}
    for vote in class_list:
        if vote not in class_cnt.keys():
            class_cnt[vote] = 0
        class_cnt[vote] += 1
    sorted_class_cnt = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True)  # 字典排序
    return sorted_class_cnt[0][0]


def create_tree(dataset, labels):
    temp_labels = copy.deepcopy(labels)
    class_list = [a[-1] for a in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return major_cnt(class_list)
    best_feat = choose_split_feature(dataset)
    best_feat_label = temp_labels[best_feat]
    my_tree = {best_feat_label: {}}
    del (temp_labels[best_feat])
    feat_values = [a[best_feat] for a in dataset]
    uniq_values = set(feat_values)
    for value in uniq_values:
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), temp_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]   # dict.keys返回一个对象，更像一个set，不支持索引
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def create_dataset():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'yes'], [1, 0, 'no'], [1, 0, 'no'], [0, 0, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


dataset, label = create_dataset()
dtree = create_tree(dataset, label)
print(dtree)
class_label = classify(dtree,label,[0,0])
print(class_label)
