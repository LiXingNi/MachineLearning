# _*_ coding:utf-8 _*_
import theano
import theano.tensor as T
from theano import function
from numpy import random as rng

# 说明:
# 建立一个对象 logistic = LogisticRg(训练数据，训练结果)
# 调用训练函数并指定训练此时 logistic.train(训练次数)
# 调用预测函数进行预测 v = logistic.predict(测试数据)

class LogisticRg:
  def __init__(self, train_data, train_result):
    self.train_data = train_data
    self.train_result = train_result
    feats = len(train_data[0])                                  # 定义特征维度
    x = T.dmatrix('x')
    w = theano.shared(rng.randn(feats), 'w')                    # w 与 b 都是共享对象,通过调整 w 与 b 使得计算结果逼近真实分类结果
    b = theano.shared(0., 'b')
    y = T.ivector('y')                                          # y 是真实分类的符号
    prob = 1 / (1 + T.exp(-T.dot(x, w) - b))                    # 线性预测模型 使用 sigmod 函数将线性结果映射到 0 - 1 区间
    self.cross = -y * T.log(prob) - (1 - y) * T.log(1 - prob)   # 代价函数使用交叉熵
    cost = self.cross.mean() + 0.01 * (w ** 2).sum()            # 使用累积误差来对参数进行更新
    self._prediction = prob > 0.5                               # 将sigmod 映射空间中小于等于0.5的归为 0
    gw, gb = T.grad(cost, [w, b])                               # 代价函数梯度下降减少预测误差
    self._train = function(
      inputs = [x, y],
      outputs = [self._prediction, self.cross],
      updates = [(w,w - 0.1 * gw),(b,b - 0.1 * gb)])            # 训练函数根据变化量更新参数
    self._predict = function([x],self._prediction)              # 预测函数

  def predict(self,x_val):
      return self._predict(x_val)                               # 进行一次预测

  def train(self,times):
      for i in range(times):
          self._train(self.train_data,self.train_result)        #使用累积误差来训练


if __name__ == "__main__":
    train_steps = 10000
    N = 400
    feats = 784                                                 #指定输入的维度，此处将维数设置的比较高，确保随机生成的数据线性可分
    D = (rng.randn(N, feats), rng.randint(0, 2, size = N))

    logistic = LogisticRg(D[0],D[1])                            # D[0] 是训练数据的矩阵，每一行是一个训练数据。D[1]是对应训练数据的分类结果
    logistic.train(train_steps)
    v = logistic.predict(D[0])
    for i in range(N):
        print D[1][i], '\t', v[i]

