# _*_ coding:utf-8 _*_
import  theano
from theano import function
import theano.tensor as T
import numpy as np
from numpy import random as Rng
import sys
sys.path.append("..")

class LogisticRg:
    def __init__(self, feature_num, x_input):
        feats = feature_num                                  # 定义特征维度
        w = theano.shared(Rng.randn(feats), 'w')                    # w 与 b 都是共享对象, 通过调整 w 与 b 使得计算结果逼近真实分类结果
        b = theano.shared(0., 'b')
        y = T.ivector('y')                                          # y 是真实分类的符号
        prob = 1 / (1 + T.exp(-T.dot(x_input, w) - b))                    # 线性预测模型 使用 sigmod 函数将线性结果映射到 0 - 1 区间
        self.cross = -y * T.log(prob) - (1 - y) * T.log(1 - prob)   # 代价函数使用交叉熵
        cost = self.cross.mean() + 0.01 * (w ** 2).sum()            # 使用累积误差来对参数进行更新
        self._prediction = prob > 0.5                               # 将sigmod 映射空间中小于等于0.5的归为 0
        gw, gb = T.grad(cost, [w, b])                               # 代价函数梯度下降减少预测误差
        self._train = function(
            inputs = [x_input, y],
            outputs = [self._prediction, self.cross],
            updates = [(w, w - 0.1 * gw), (b, b - 0.1 * gb)])            # 训练函数根据变化量更新参数
        self.params = [w, b]
        self._predict = function([x_input], self._prediction)              # 预测函数

    def predict(self, x_val):
        return self._predict(x_val)                               # 进行一次预测

    def train(self, times, train_data, train_result):
        for i in range(times):
            self._train(train_data, train_result)        #使用累积误差来训练

class Boost(object):
    def __init__(self, x, y, row, col,symbol_D_t):
        # x 是 fmatrix 符号类型
        # y 是 ivector 符号类型
        # row 是训练数据的数量
        # col 是训练数据的特征数量
        self.D = symbol_D_t # self.D_t 是 fcol 类型，是按列传播的

        # 问题**********： 如果 self.D 是ivector，则 ivector 无法给 x 执行加权
        #***************: 如果 self.D 是 icol，则 icol * ivector是矩阵，在算 n_D_t 的时候不能与ivector相乘

        self.classifier = LogisticRg(col, x)  #特征数量
        self.y_prediction = self.classifier._prediction
        self.prob_error = T.sum(T.neq(self.y_prediction, y)) / row
        self.alpha_t = 0.5 * np.log2((1 - self.prob_error) / self.prob_error)
        self.n_D_t = self.D * T.exp2(-self.alpha_t * self.y_prediction * y) / (
            T.sum(self.D * T.exp2(-self.alpha_t * self.y_prediction * y))
        )
        self.boost = function([x, y, self.D], outputs = [self.alpha_t, self.n_D_t]) # 计算权重与下一个分布函数
        self.predict = self.classifier.predict #实际预测函数

        self.reWeightData = function([x, self.D], self.D * x)

    def train_model(self, D_t, times, input_data, input_result):
        #将当前分布对训练数据进行重新赋值权重
        train_data = self.reWeightData(input_data, D_t)
        self.classifier.train(times, train_data, input_result)


class AdaBoostring(object):
    def __init__(self, n_level, row, col):
        # row 是训练数据的数量
        # col 是训练数据的特征数量

        x = T.fmatrix('x')
        y = T.ivector('y')

        self.boosting_list = []
        for i in range(n_level):

            if i == 0:
                D_t = T.fcol('D_t')
            else:
                D_t = self.boosting_list[-1].n_D_t

            self.boosting_list.append(Boost(x, y, row, col, D_t))




















