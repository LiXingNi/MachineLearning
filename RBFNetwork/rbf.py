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

def loadFile(file_name):
    f_obj = open(file_name)
    line = f_obj.readline()  #读出第一行标识行

    i_dim,o_dim= line.strip().split('\t')
    i_dim = int(i_dim)
    o_dim = int(o_dim)

    train_data = []
    train_label = []

    for line in f_obj:
        lis = [int(s) for s in line.strip().split('\t')]
        train_data.append(lis[0:i_dim])
        train_label.append(lis[i_dim])
    return train_data,train_label

class RBF:
    def __init__(self,h_num,train_data,train_label,o_num,eta,center_set = None):
        self._eta = eta
        self._h_num = h_num
        self._center_set = center_set
        self._o_dim= o_num
        self._i_dim= len(train_data[0])
        self._train_data = train_data[:]

        #根据标签的值和输出的维度构造输出集合
        self._train_result = []
        for label in train_label:
            l = [0] * o_num
            l[label] = 1
            self._train_result.append(l)

        self._h_node = []
        self._o_node = []
        self._initHiddenNode()     #初始化隐层顶点
        self._initOutputNode()       #初始化输出层顶点


    def _orderChooseCenterPoint(self):
        #随机选样本中的h_num作为中心点
        center_set = []
        center_range = (0,len(self._train_data) - 1)
        for i in range(self._h_num):
            center_set.append(self._train_data[i])
        return center_set

    def _initHiddenNode(self):
        center_set = self._center_set
        if center_set == None:
            center_set = self._orderChooseCenterPoint()
        #for c in center_set:
        #    print(c)

        #求点之间的最大距离
        center_pair = [[x,y] for x in center_set for y in center_set]
        max_distance = np.linalg.norm(max(center_pair,key = lambda k: np.linalg.norm(np.array(k[0])- np.array(k[1]))))

        #初始化隐层顶点
        delta = max_distance / np.sqrt(2 * self._h_num)
        self._h_node = [HNode(center_set[i],delta) for i in range(self._h_num)]

    def _initOutputNode(self):
        self._o_node = [ONode(self._h_num) for i in range(self._o_dim)]


    #根据输入返回训练系统的输出
    def predict(self,input):
        return self.rbfCount(self.hiddenOutput(input))

    #输入隐层输出，返回神经网络预测值
    def rbfCount(self,h_output):
        y_output = [o_node.calcOutput(h_output) for o_node in self._o_node]
        return y_output

    #输入训练数据，输出隐层输出
    def hiddenOutput(self,input):
        h_output = [RBFunction(input,h_node._center,h_node._delta) for h_node in self._h_node]
        return h_output


    def train(self,times):
        for i in  range(times):
            #if i % 10 == 0:
            print("train:%d"%(i,))
            self._trains()

    #新的矩阵运算
    def _trains(self):
        x_matrix = np.array(self._train_data)
        y_real_matrix = np.array(self._train_result)
        green_matrix = [self.hiddenOutput(input) for input in self._train_data]
        y_predict_matrix = np.array([self.rbfCount(h_output) for h_output in green_matrix])
        green_matrix = np.array(green_matrix)

        c_matrix = np.array([h_node._center for h_node in self._h_node])
        w_matrix = np.array([o_node._weight for o_node in self._o_node])

        x_c_matrix = x_matrix - c_matrix
        x_c_2_matrix = [np.linalg.norm(dif)**2 for dif in x_c_matrix]
        e_y = y_predict_matrix - y_real_matrix

        for j in range(self._h_num):
            w_y_sum = [sum(np.array(w_matrix[j] * np.array(y))) for y in e_y]
            delta_c_j = green_matrix[j] * x_c_matrix
            delta_c_j = np.dot(delta_c_j,w_y_sum)



    #根据所有输入文件进行训练
    def _train(self):
        for index in range(len(self._train_data)):
            #if index % 100 == 0:
            #    print("\t%d"%(index))
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
    train_data,train_label = loadFile("train.txt")
    rbf = RBF(4,train_data,train_label,2,0.2,None)
    rbf.train(1000)
    test = [[0,1],[1,0],[0,0],[1,1]]
    for t in test:
        n,y = rbf.rbfCount(t)
        print(t," ",y)



