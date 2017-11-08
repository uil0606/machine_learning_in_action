from numpy import *


def load_dataset():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'i', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['my', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocal_list(dataset):
    # 出现的单词列表
    vocal_set = set([])
    for docu in dataset:
        vocal_set = vocal_set | set(docu)
    return list(vocal_set)


def word_to_vec(vocal_list, input_set):
    # 将输入的句子转换成单词列表向量
    return_vec = [0] * len(vocal_list)
    for word in input_set:
        if word in vocal_list:
            return_vec[vocal_list.index(word)] += 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return return_vec


def train_NB0(train_mat, train_category):
    # 输入训练集对应单词集矩阵
    num_train_docs = len(train_mat)
    num_words = len(train_mat[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_demon = 2
    p1_demon = 2
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_mat[i]
            p1_demon += sum(train_mat[i])
        else:
            p0_num += train_mat[i]
            p0_demon += sum(train_mat[i])
    p1_vec = log(p1_num / p1_demon)
    p0_vec = log(p0_num / p0_demon)
    return p0_vec, p1_vec, p_abusive


def classify_NB(vec_to_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec_to_classify * p1_vec) + log(p_class1)
    p0 = sum(vec_to_classify * p0_vec) + log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def test_NB():
    list_posts, list_classes = load_dataset()
    my_vocal_list = create_vocal_list(list_posts)
    train_mat = []
    for post in list_posts:
        train_mat.append(word_to_vec(my_vocal_list, post))
    p0_v, p1_v, p_ab = train_NB0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(word_to_vec(my_vocal_list, test_entry))
    print(test_entry, 'classified as: ', classify_NB(this_doc, p0_v, p1_v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = array(word_to_vec(my_vocal_list, test_entry))
    print(test_entry, 'classified as: ', classify_NB(this_doc, p0_v, p1_v, p_ab))


# test_NB()

def text_parse(big_str):
    import re
    list_tokens = re.split(r'\w*', big_str)
    return [tok.low() for tok in list_tokens if len(tok) > 2]  # 小写化长度大于2的单词，作为输入


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocal_list = create_vocal_list(doc_list)
    training_set = range(50)
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])  # 生成测试集
        del (training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(word_to_vec(vocal_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_NB0(array(train_mat), array(train_classes))
    err_count = 0
    for doc_index in test_set:
        word_vec = word_to_vec(vocal_list, doc_list[doc_index])
        if classify_NB(array(word_vec), p0_v, p1_v, p_spam) != class_list[doc_index]
            err_count += 1
    print('the error rate is: ', float(err_count) / len(test_set))
