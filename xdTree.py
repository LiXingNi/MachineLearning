from math import log
import operator


def createDataSet(file_name):
    fobj = open(file_name,'r')

    line = fobj.readline()
    feature_attr = (line.strip()).split('\t')

    data_set = []
    for line in fobj:
        data_set.append(line.strip().split('\t'))

    return (feature_attr,data_set)
    
def splitDataSet(data_set,axis,value):
    sub_data_set = []

    for line in data_set:
        if line[axis] == value:
            sub_line = line[:axis]
            sub_line.extend(line[axis+1:])
            sub_data_set.append(sub_line)
            
    return sub_data_set

def majorityValue(data_set):
    value_dict = {}

    for line in data_set:
        if line[-1] not in value_dict.keys():
            value_dict[line[-1]] = 0
        value_dict[line[-1]] += 1

    new_dict = sorted(value_dict.items(),key = operator.itemgetter(1),reverse = True)

    return new_dict[0][0]



############################################### gini

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

        if curr_gain < best_gain:
            best_gain = curr_gain
            best_feature = axis


    return best_feature
    



############################################ shannon

def calShannonEntropy(dataSet):
    setNum = len(dataSet)
    valueDct = {}
    entropy = 0.0

    for featVec in dataSet:
        currentVal = featVec[-1]
        if currentVal not in valueDct.keys():
            valueDct[currentVal] = 0
        valueDct[currentVal] += 1

    for value in valueDct:
        prob = float(valueDct[value]) / setNum
        entropy -= prob * log(prob,2)

    return entropy


def chooseBestFeatureToSplit(dataSet):
    featureNum = len(dataSet[0]) - 1
    dataSetNum = len(dataSet)
    bestEntropy = 0.0;bestFeature = -1
    baseEntropy = calShannonEntropy(dataSet)

    for axis in range(featureNum):
        featValLis = [line[axis] for line in dataSet]
        featValSet = set(featValLis)

        currEntropy = 0.0
        
        for featVal in featValSet:
            subDataSet = splitDataSet(dataSet,axis,featVal)
            prob = float(len(subDataSet)) / dataSetNum

            currEntropy += prob * calShannonEntropy(subDataSet)

        infoEntropy = baseEntropy - currEntropy
        if(infoEntropy > bestEntropy):
            bestEntropy = infoEntropy
            bestFeature = axis

    return bestFeature



#################################################



class TreeNode:
    def __init__(self,feature_str):
        self.feature = feature_str
        self.children_dic = {}


def createTree(data_set,feature_attr,feature_dic,tree_func):
    if [line[-1] for line in data_set].count(data_set[0][-1]) == len(data_set):
        return TreeNode(data_set[0][-1])

    if len(feature_attr) == 0:
        return TreeNode(majorityValue(data_set))

    best_feature_axis = tree_func(data_set)
    best_feature_attr = feature_attr[best_feature_axis]
    best_feature_value = feature_dic[best_feature_attr]

    del(feature_attr[best_feature_axis])
    
    curr_node = TreeNode(best_feature_attr)

    for value in best_feature_value:
        sub_data_set = splitDataSet(data_set,best_feature_axis,value)
        
        if len(sub_data_set) == 0:
            curr_node.children_dic[value] = TreeNode(majorityValue(data_set))      
        else:
            sub_feature_attr = feature_attr[:]
            curr_node.children_dic[value] = createTree(sub_data_set,sub_feature_attr,feature_dic,tree_func)

    return curr_node    


def printTree(tree_node):
    print(tree_node.feature)
    if len(tree_node.children_dic) == 0:
        return
    for key in tree_node.children_dic:
        print(key)
        printTree(tree_node.children_dic[key])


class Tree:

    def __init__(self,file_name,tree_func = chooseBestFeatureGINI):
        self.feature_attr,self.data_set = createDataSet(file_name)

        self.feature_dic = {}
        for axis in range(len(self.feature_attr)):
            self.feature_dic[self.feature_attr[axis]] = list(set([line[axis] for line in self.data_set]))

        self.root = createTree(self.data_set,self.feature_attr,self.feature_dic, tree_func)


    def checkData(self):
        for key in self.feature_dic:
            print(key,self.feature_dic[key])

        print(self.feature_attr)
        
        for i in self.data_set:
            print(i)

    def printTree(self):
        printTree(self.root)



if __name__ == "__main__":
    tree = Tree('2_0.txt')
    tree.printTree()
        
        
