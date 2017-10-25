import numpy as np
import operator
import time
from os import listdir


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 长度
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 重复多次构成矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # argsort
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def img2vector(file):
    returnVec = np.zeros((1, 1024))
    with open(file) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVec[0, 32 * i + j] = int(lineStr[j])
        return returnVec


def read_and_convert(filePath):
    dataLabel = []
    fileList = listdir(filePath)
    fileAmount = len(fileList)
    dataMat = np.zeros((fileAmount, 1024))
    for i in range(fileAmount):
        fileNameStr = fileList[i]
        classTag = int(fileNameStr.split(".")[0].split("_")[0])
        dataLabel.append(classTag)
        dataMat[i, :] = img2vector(filePath + "/" + fileNameStr)
    return dataMat, dataLabel


group, labels = read_and_convert(
    'F:\\optdigits\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\digits\\trainingDigits')

test_group, test_labels = read_and_convert(
    'F:\\optdigits\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\digits\\testDigits')

# print(classify0(test_group[100], group, labels, 51))

test_pre_fal={}
test_pre_cor={}
for i in range(len(test_group)):
    pre=(classify0(test_group[i], group, labels, 3))
    if pre!=test_labels[i]:
        test_pre_fal[pre]=test_pre_fal.get(pre,0)+1
    else:
        test_pre_cor[pre]=test_pre_cor.get(pre,0)+1

print(sorted(test_pre_cor.items(), key=operator.itemgetter(0)))
print(sorted(test_pre_fal.items(), key=operator.itemgetter(0)))
