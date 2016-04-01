import random
import numpy as np
from functools import reduce

def distance(arr1,arr2):
    return np.linalg.norm(np.array(arr1)-np.array(arr2))

def RBFunction(x_,center_,delta):  # X 与 center 是 list类型
    e = np.e
    mem = distance(x_,center_)
    return e**(-(mem**2)/(2*(delta ** 2)))


class HNode:
    def __init__(self,center,delta):
        self._center = center[:]
        self._delta = delta

class ONode:
    def __init__(self,h_num):
        self._weight = []
        for i in range(h_num):
            self._weight.append(random.random())

    def calcOutput(self,input):
        return reduce(lambda x,y: x + y,map(lambda x,y : x * y,input,self._weight))

class RBF:
    def __init__(self,h_num,file_name,eta):
        self._eta = eta
        self._h_num = h_num
        self._o_dim= 0
        self._i_dim= 0
        self._train_data = []
        self._train_result = []
        self._h_node = []
        self._o_node = []
        self.loadFile(file_name)  #读取输入输出序列
        self._initHiddenNode()     #初始化隐层顶点
        self._initOutputNode()       #初始化输出层顶点


    def loadFile(self,file_name):
        f_obj = open(file_name)
        line = f_obj.readline()  #读出第一行标识行

        self._i_dim,self._o_dim= line.strip().split('\t')
        self._i_dim = int(self._i_dim)
        self._o_dim = int(self._o_dim)

        for line in f_obj:
            lis = [int(s) for s in line.strip().split('\t')]
            self._train_data.append(lis[0:self._i_dim])
            self._train_result.append(lis[self._i_dim:])

    def _initHiddenNode(self):
        #随机选样本中的h_num作为中心点
        center_set = []
        center_range = (0,len(self._train_data) - 1)
        for i in range(self._h_num):
            center_set.append(self._train_data[i])
        #求点之间的最大距离
        center_pair = [[x,y] for x in center_set for y in center_set]
        max_distance = np.linalg.norm(max(center_pair,key = lambda k: np.linalg.norm(np.array(k[0])- np.array(k[1]))))

        #初始化隐层顶点
        delta = max_distance / np.sqrt(2 * self._h_num)
        for i in range(self._h_num):
            self._h_node.append(HNode(center_set[i],delta))

    def _initOutputNode(self):
        for i in range(self._o_dim):
            self._o_node.append(ONode(self._h_num))


    #根据输入返回训练系统的输出
    def rbfCount(self,input):
        h_output = []
        for i,h_node in enumerate(self._h_node):
            #计算每个隐层顶点的输出
            h_node = self._h_node[i]
            h_output.append(RBFunction(input,h_node._center,h_node._delta))

        #根据每个输出顶点的权重计算每个输出顶点的值
        y_output = []
        for i, o_node in enumerate(self._o_node):
            y_output.append(o_node.calcOutput(h_output))
        return h_output,y_output

    def train(self,times):
        for i in  range(times):
            self._train()

    #根据所有输入文件进行训练
    def _train(self):
        for index in range(len(self._train_data)):
            x = self._train_data[index]
            h_output,y_predict = self.rbfCount(x)
            y_actual = self._train_result[index]

            #计算 e_h的值
            e_h = list(np.array(y_actual) - np.array(y_predict))

            #根据预测值调整隐层
            self._adjustHNode(index,y_predict,e_h,h_output)

            #根据预测值调整输出层的权重
            self._adjustONode(index,y_predict,e_h,h_output)

    def _adjustONode(self,index,y_predict,e_h,h_output):
        for h,o_node in enumerate(self._o_node):
            for j in range(self._h_num):
                o_node._weight[j] += self._eta * e_h[h]* h_output[j]

    def _adjustHNode(self,index,y_predict,e_h,h_output):
        x = self._train_data[index]
        y_actual = self._train_result[index]

        for j,h_node in enumerate(self._h_node):
        #逐个调整隐层顶点

            #计算叠加的 w_jh * e_h
            val = 0
            for h in range(self._o_dim):
               val += self._o_node[h]._weight[j] * e_h[h]

            #计算cj 调整量
            delta_cj_np = (self._eta / (h_node._delta ** 2)) * h_output[j] * (np.array(x) - np.array(h_node._center))
            delta_cj_np *= val

            #计算 delta 调整量
            delta_delta = (self._eta / (h_node._delta**3)) * (distance(x,h_node._center)**2) * h_output[j]
            delta_delta *= val

            #调整隐层中心点值
            h_node._center = list(np.array(h_node._center + delta_cj_np))
            h_node._delta += delta_delta




if __name__ == "__main__":
    rbf = RBF(4,"train.txt",0.2)
    rbf.train(1000)
    test = [[0,1],[1,0],[0,0],[1,1]]
    for t in test:
        n,y = rbf.rbfCount(t)
        print(t," ",y)



