from math import log
import operator
from functools import reduce



G_Accuracy = 1e-6

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

        if best_gain - curr_gain > G_Accuracy:
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


def chooseBestFeatureShannon(dataSet):
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
        if infoEntropy - bestEntropy > G_Accuracy:
            #print('*********',infoEntropy,'\t',bestEntropy)
            bestEntropy = infoEntropy
            bestFeature = axis

    
    return bestFeature



#################################################



class TreeNode:
    def __init__(self,feature_str):
        self.feature = feature_str
        self.children_dic = {}
        
##########################################################  递归辅助函数

def createTree(data_set,feature_attr,feature_dic,tree_func):         # 需要递归调用，因此不能写为类内函数
    if [line[-1] for line in data_set].count(data_set[0][-1]) == len(data_set):
        return TreeNode(data_set[0][-1])

    if len(feature_attr) == 0:
        return TreeNode(majorityValue(data_set))

    best_feature_axis = tree_func(data_set)
    best_feature_attr = feature_attr[best_feature_axis]
    best_feature_value = feature_dic[best_feature_attr]

    #print(best_feature_attr)

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


def printTree(tree_node):      # 需要递归调用，因此不能写为类内函数
    print(tree_node.feature)
    if len(tree_node.children_dic) == 0:
        return
    for key in tree_node.children_dic:
        print(key)
        printTree(tree_node.children_dic[key])


def getNumLeafs(tree_node):
    if len(tree_node.children_dic) == 0:
        return 1
    return reduce(lambda x,y:x+y,map(getNumLeafs,tree_node.children_dic.values()))



def getTreeDepth(tree_node):
    if len(tree_node.children_dic) == 0:
        return 1
    return reduce(lambda x,y : max(x,y),map(lambda x:1+getTreeDepth(x),tree_node.children_dic.values()))

################################################################################

class Tree:

    def __init__(self,file_name,tree_func = chooseBestFeatureGINI):
        self.feature_attr,self.data_set = createDataSet(file_name)  # feature_attr 是特征属性名称的列表
        self.feature_dic = {}    # feature_dic 是特征属性名称对应的所有可能取值

        
        for axis in range(len(self.feature_attr)):  # 获取每个特征属性名称对应的所有可能取值
            self.feature_dic[self.feature_attr[axis]] = list(set([line[axis] for line in self.data_set]))

        feature_attr = self.feature_attr[:]
        self.root = createTree(self.data_set,feature_attr,self.feature_dic, tree_func)


    def checkData(self):  # 辅助函数，用来输出调试必要信息
        for key in self.feature_dic:
            print(key,self.feature_dic[key])

        print(self.feature_attr)
        
        for i in self.data_set:
            print(i)

    def printTree(self):
        printTree(self.root)

    def getNumLeafs(self):
        return getNumLeafs(self.root)

    def getTreeDepth(self):
        return getTreeDepth(self.root)


class TestTree:
    def __init__(self,file_name,tree):
        self.tree = tree
        self.test_seq = self.readTestData(file_name)
        self.result = self.countAccuracy()
        

    def getClassicResult(self,feature_vec):  # 对一个属性向量的求解
        tree_node = self.tree.root
        while len(tree_node.children_dic) != 0:           # 当前结点不是叶节点
            axis = self.tree.feature_attr.index(tree_node.feature)   # 当前节点特征属性名称对应的axis
            tree_node = tree_node.children_dic[feature_vec[axis]] # 取出当前特征向量中对应特征的值，取出该值对应的分支，作为新的结点
        return tree_node.feature


    def readTestData(self,file_name):   # 从测试数据的文件名读出测试数据，[[],[],]。其中每一行为一个特征向量，该行最后一列为真实分类值
        fobj = open(file_name,'r')
        test_lis = []
        for line in fobj:
            lis = (line.strip()).split('\t')
            test_lis.append(lis)
        return test_lis


    def countAccuracy(self):
        right_num = 0
        for line in self.test_seq:
            res = self.getClassicResult(line)
            print(line,res)
            if res == line[-1]:
                right_num += 1
        return float(right_num) / len(self.test_seq)
            

if __name__ == "__main__":
    tree = Tree('4_2_train.txt',chooseBestFeatureShannon)
    #tree = Tree('4_2_train.txt')
    #tree = Tree('2_0.txt',chooseBestFeatureShannon)
    tree.printTree()

    test_tree = TestTree('4_2_test.txt',tree)
    print(test_tree.result)
    print(tree.getNumLeafs())
    print(tree.getTreeDepth())
    
