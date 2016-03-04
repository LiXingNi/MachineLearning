from decisionTree.helpFunc import *


def giniIndex(data_set):
    value_dic = {}
    for line in data_set:
        clasic = line[-1]
        if clasic not in value_dic.keys():
            value_dic[clasic] = 0
        value_dic[clasic] += 1

    total_sum = 0
    for key in value_dic:
        prob = float(value_dic[key]) / len(data_set)
        total_sum += prob ** 2

    return 1 - total_sum



def chooseBestFeatureGINI(data_set):
    feature_num = len(data_set[0]) - 1
    data_num = len(data_set)

    best_feature = -1
    best_gain = 1.0

    for axis in range(feature_num):
        axis_fv_lis = [line[axis] for line in data_set]
        axis_fv = set(axis_fv_lis)
        curr_gain = 0.0

        for value in axis_fv:
            sub_data_set = splitDataSet(data_set,axis,value)
            curr_gain += float(len(sub_data_set)) / data_num * giniIndex(sub_data_set)

        if best_gain - curr_gain > G_Accuracy:
            best_gain = curr_gain
            best_feature = axis


    return best_feature



