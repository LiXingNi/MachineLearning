from math import e
import random

def sigmoid(x):
    return 1.0 / (1 + e**(-x))

class Neuron:
    def __init__(self,func):
        self.weight = []                #当前神经元各个输入分支的权重
        self.threshold = random.random()#当前神经元的权重
        self.activeFunc = func          #当前神经元的激活函数

    def initThreshold(self,input_num):
        for i in range(input_num):
            self.weight.append(random.random())


    def calOutput(self,input_vec):      # 输入向量
        input_val = 0.0
        for i in range(len(input_vec)):
            input_val += self.weight[i] * input_vec[i]
        input_val -= self.threshold
        #print(input_val)
        return self.activeFunc(input_val)


class NeuronNetwork:
    def __init__(self,ip_layer_num,hd_layer_num,op_layer_num,func,l_rate = 0.2):
        self.l_rate = l_rate
        self.hd_layer = []
        self.op_layer = []

        # init hidden layer
        for i in range(hd_layer_num):
            neuron = Neuron(func)
            #neuron.initThreshold(ip_layer_num,self.weight_init)
            neuron.initThreshold(ip_layer_num)
            self.hd_layer.append(neuron)

        #init output layer
        for i in range(op_layer_num):
            neuron = Neuron(func)
            neuron.initThreshold(hd_layer_num)
            self.op_layer.append(neuron)


    def calLayerOutput(self,input_vector,hd_layer):             # 计算隐层（hd_layer:True）或输出层神经元的输出向量
        #遍历隐层的每个神经元计算输出，并将输出存储为隐层的输出向量
        output_lis = []
        if hd_layer == True:
            for i in range(len(self.hd_layer)):
                output_lis.append(self.hd_layer[i].calOutput(input_vector))
        else:
            for i in range(len(self.op_layer)):
                output_lis.append(self.op_layer[i].calOutput(input_vector))
        return output_lis


    def _calOutput(self,input_vector):  #测试调用，测试需要隐层输出与输出层输出
        hd_layer_output_vector = self.calLayerOutput(input_vector,True)
        #print(hd_layer_output_vector)
        op_layer_output_vector = self.calLayerOutput(hd_layer_output_vector,False)
        return hd_layer_output_vector,op_layer_output_vector

    def calOutput(self,input_vector):  # 给用户调用，用户只关心输出层输出
        hd_layer_output,op_layer_output = self._calOutput(input_vector)
        return op_layer_output

    def _readData(self,file_name):
        fobj = open(file_name,'r')

        line = fobj.readline() # 数据集第一行存储了：输入是几维的，输出是几维的

        i_num,o_num = line.strip().split('\t')
        i_num = int(i_num)
        o_num = int(o_num)
        train_vec = []
        train_res = []
        for line in fobj:
            lis = [int(s) for s in line.strip().split('\t')]
            train_vec.append(lis[0:i_num])
            train_res.append(lis[i_num:i_num + o_num])

        return train_vec,train_res        # train_vec 存的是测试向量集合， train_res存的是测试向量结果集合




    def train(self,file_name,times):
        train_vec,train_res = self._readData(file_name)
        for i in range(times):
            for j in range(len(train_vec)):
                data_vec = train_vec[j]
                data_res = train_res[j]
                hd_layer_o,op_layer_o = self._calOutput(data_vec)
                self.modifyParam(data_vec,hd_layer_o,op_layer_o,data_res)

        for j in range((len(train_vec))):
            data_vec = train_vec[j]
            data_res = train_res[j]
            y = self.calOutput(data_vec)
            print(data_vec, end='\t')
            print(y,'\t',data_res)



    def _calGj(self,p_op_layer_output,r_op_layer_output):
        g_lis = []
        for j in range(len(self.op_layer)):
            r_y_j = r_op_layer_output[j]
            p_y_j = p_op_layer_output[j]
            g_j = p_y_j * (1 - p_y_j)*(r_y_j - p_y_j)
            g_lis.append(g_j)
        return g_lis

    def _calEh(self,g_lis,hd_layer_output):
        e_lis = []
        for h in range(len(self.hd_layer)):
            # 计算求和式
            e_h = 0.0

            for j in range(len(self.op_layer)):
                e_h += self.op_layer[j].weight[h] * g_lis[j]

            b_h = hd_layer_output[h]
            e_lis.append(b_h * (1 - b_h) * e_h)
        return e_lis

    #标准BP神经网络
    def _calModifyHdParam(self,ip_layer_output,e_lis):
        threshold_modify = []
        weight_modify = []

        for h in range(len(self.hd_layer)):
            neuron = self.hd_layer[h]
            #修改阈值
            threshold = -self.l_rate * e_lis[h]
            # 将修改值加入修改集合
            threshold_modify.append(threshold)

            # 修改神经元的输入权重集合
            weight_h = []
            for i in range(len(neuron.weight)):
                v_weight = self.l_rate * e_lis[h] * ip_layer_output[i]
                weight_h.append(v_weight)

            weight_modify.append(weight_h)

        return (threshold_modify,weight_modify)

    def _calModifyOpParam(self,hd_layer_output,g_lis):
        threshold_modify = []
        weight_modify = []

        for j in range(len(self.op_layer)):
            neuron = self.op_layer[j]
            #修改阈值
            threshold = -self.l_rate * g_lis[j]
            threshold_modify.append(threshold)

            #修改神经元的权重集合
            weight_j = []
            for h in range(len(neuron.weight)):
                w_weight = self.l_rate * g_lis[j] * hd_layer_output[h]
                weight_j.append(w_weight)
            weight_modify.append(weight_j)
        return (threshold_modify,weight_modify)

    def _modifyParam(self,hd_t_m,hd_w_m,op_t_m,op_w_m):
        #修改隐层
        for h in range(len(self.hd_layer)):
            neuron = self.hd_layer[h]
            neuron.threshold += hd_t_m[h]

            for i in range(len(neuron.weight)):
                neuron.weight[i] += hd_w_m[h][i]

        #修改输出层
        for j in range(len(self.op_layer)):
            neuron = self.op_layer[j]
            neuron.threshold += op_t_m[j]

            for h in range(len(neuron.weight)):
                neuron.weight[h] += op_w_m[j][h]

    def modifyParam(self,ip_layer_output,hd_layer_output,op_layer_output,r_op_layer_output,standard_BP = True):
        #根据输出计算出gj
        g_lis = self._calGj(op_layer_output,r_op_layer_output)
        e_lis = self._calEh(g_lis,hd_layer_output)

        #调整隐层的权重与阈值：
        hd_threshold_m,hd_weight_m = self._calModifyHdParam(ip_layer_output,e_lis)

        #调整输出层的权重与阈值
        op_threshold_m,op_weight_m = self._calModifyOpParam(hd_layer_output,g_lis)

        if standard_BP == True:
            self._modifyParam(hd_threshold_m,hd_weight_m,op_threshold_m,op_weight_m)

        return hd_threshold_m,hd_weight_m,op_threshold_m,op_weight_m


if __name__ == "main":
    neuronN = NeuronNetwork(2,3,1,sigmoid)
    neuronN.train('train.txt',10000)

