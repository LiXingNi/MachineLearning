from drawTree import *
from giniTree import chooseBestFeatureGINI
from shannonDTree import chooseBestFeatureShannon

from decisionTree.helpFunc import *


class TreeNode:
    def __init__(self,feature_str,father,path):
        self.father = father
        self.path = path
        self.feature = feature_str
        self.children_dic = {}

    def isLeaf(self):
        if len(self.children_dic) == 0:
            return True
        else:
            return False
        
##########################################################  递归辅助函数

def createTree(father_node,path_value,data_set,feature_attr,feature_dic,tree_func):         # 需要递归调用，因此不能写为类内函数
    if [line[-1] for line in data_set].count(data_set[0][-1]) == len(data_set):
        return TreeNode(data_set[0][-1],father_node,path_value)

    if len(feature_attr) == 0:
        return TreeNode(majorityValue(data_set),father_node,path_value)

    best_feature_axis = tree_func(data_set)
    best_feature_attr = feature_attr[best_feature_axis]
    best_feature_value = feature_dic[best_feature_attr]

    #print(best_feature_attr)

    del(feature_attr[best_feature_axis])
    
    curr_node = TreeNode(best_feature_attr,father_node,path_value)       # tree_node 中feature 属性存储的特征属性的名称

    for value in best_feature_value:
        sub_data_set = splitDataSet(data_set,best_feature_axis,value)
        
        if len(sub_data_set) == 0:
            curr_node.children_dic[value] = TreeNode(majorityValue(data_set),curr_node,value)
        else:
            sub_feature_attr = feature_attr[:]
            curr_node.children_dic[value] = createTree(curr_node,value,sub_data_set,sub_feature_attr,feature_dic,tree_func)

    return curr_node    


def printTree(tree_node):      # 需要递归调用，因此不能写为类内函数
    print(tree_node.feature)
    if len(tree_node.children_dic) == 0:
        return
    for key in tree_node.children_dic:
        print(key)
        printTree(tree_node.children_dic[key])



################################################################################

class Tree:

    def __init__(self,file_name,tree_func = chooseBestFeatureGINI):
        self.feature_attr,self.data_train = createDataSet(file_name)  # feature_attr 是特征属性名称的列表
        self.feature_dic = {}    # feature_dic 是特征属性名称对应的所有可能取值
        self.BFS_lis = []
        self.data_test = []
        
        for axis in range(len(self.feature_attr)):  # 获取每个特征属性名称对应的所有可能取值
            self.feature_dic[self.feature_attr[axis]] = list(set([line[axis] for line in self.data_train]))

        feature_attr = self.feature_attr[:]
        self.root = createTree(None,None,self.data_train,feature_attr,self.feature_dic, tree_func)


    def checkData(self):  # 辅助函数，用来输出调试必要信息
        for key in self.feature_dic:
            print(key,self.feature_dic[key])

        print(self.feature_attr)
        
        for i in self.data_train:
            print(i)

    def printTree(self):
        printTree(self.root)

    def getNumLeafs(self):
        return getNumLeafs(self.root)

    def getTreeDepth(self):
        return getTreeDepth(self.root)

    def predict(self,feature_vec):  # 对一个属性向量的求解
        tree_node = self.root
        while len(tree_node.children_dic) != 0:           # 当前结点不是叶节点
            axis = self.feature_attr.index(tree_node.feature)   # 当前节点特征属性名称对应的axis
            tree_node = tree_node.children_dic[feature_vec[axis]] # 取出当前特征向量中对应特征的值，取出该值对应的分支，作为新的结点
        return tree_node.feature

    def readTestData(self,file_name):
        fobj = open(file_name,'r')
        self.data_test = []
        for line in fobj:
            lis = (line.strip()).split('\t')
            self.data_test.append(lis)

    def innerCalAccuracy(self):
        right_num = 0
        for line in self.data_test:
            res = self.predict(line)
            print(line,res)
            if res == line[-1]:
                right_num += 1
        return float(right_num) / len(self.data_test)

    def calAccuracy(self,file_name):
        self.readTestData(file_name)
        return self.innerCalAccuracy()

    def BFSTree(self):
        self.BFS_lis = [self.root,] # 从根节点开始遍历
        i = 0
        while i != len(self.BFS_lis):
            curr_node = self.BFS_lis[i]
            #if curr_node.father is not None:       #输出BFS结果时去掉注释即可
            #    print(curr_node.path)
            #print(curr_node.feature)
            for key in curr_node.children_dic:                      # 增加到当前List 之后
                if not curr_node.children_dic[key].isLeaf():
                    self.BFS_lis.append(curr_node.children_dic[key])
            i+=1
        #for node in self.BFS_lis:
        #    print(node.feature)

    def findPath(self,tmp_node):
          # 找到当前支路的path
        pathway_feature_attrval = {}
        while tmp_node.father is not None:
            pathway_feature_attrval[tmp_node.father.feature] = tmp_node.path   #字典中存的是：属性->属性值
            tmp_node = tmp_node.father
        return pathway_feature_attrval

    def postPruning(self):
        self.BFSTree()

        node_axis = len(self.BFS_lis) -1
        while node_axis >= 0:
            curr_node = self.BFS_lis[node_axis]  # 从后往前检测非叶节点是否有存在的必要
            node_axis -= 1

            #当前精度
            curr_accuracy = self.innerCalAccuracy()

            # 根据当前节点的路径字典集筛选出合适的数据集
            path_dic = self.findPath(curr_node)
            curr_data_set = self.data_train[:]

            for key in path_dic:
                index = self.feature_attr.index(key)
                curr_data_set = [line for line in curr_data_set if line[index] == path_dic[key]]

            # 根据当前数据集算出替换的分类节点，并进行替换后计算精度
            new_node = TreeNode(majorityValue(curr_data_set),curr_node.father,curr_node.path)

            if curr_node.father is not None:  #当前节点不是根节点
                curr_node.father.children_dic[curr_node.path] = new_node
            else:
                self.root = new_node

            new_accuracy = self.innerCalAccuracy()

            #若精度提升，则删除节点，否则，不删除
            if new_accuracy - curr_accuracy < G_Accuracy:
                if curr_node.father is not None:
                    curr_node.father.children_dic[curr_node.path] = curr_node
                else:
                    self.root = curr_node

            print("curr:",curr_accuracy,'\tnew:',new_accuracy)






if __name__ == "__main__":
    tree = Tree('4_2_train.txt',chooseBestFeatureShannon)
    #tree.printTree()
    print(tree.calAccuracy('4_2_test.txt'))

    #createPlot(tree.root)

    tree.postPruning()
    print(tree.innerCalAccuracy())

    createPlot(tree.root)


