# _*_ coding:utf-8 _*_
from __future__ import division
import theano
from theano import tensor as T
from theano import function
import numpy as np
import gzip
import cPickle as pickle
import timeit

def loadData(dataset):
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)
    def sharedData(data_xy):
        data_x, data_y = data_xy
        share_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX), borrow = True)
        share_y = theano.shared(np.asarray(data_y, dtype = theano.config.floatX), borrow = True)
        return share_x, T.cast(share_y, 'int32')

    train_set_x, train_set_y = sharedData(train_set)
    valid_set_x, valid_set_y = sharedData(valid_set)
    test_set_x, test_set_y = sharedData(test_set)

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)


class Regression(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
        self.W = theano.shared(np.zeros((n_in, n_out), dtype = theano.config.floatX), borrow = True)
        self.b = theano.shared(np.zeros((n_out, ), dtype = theano.config.floatX), borrow = True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.pred = T.argmax(self.p_y_given_x, axis = 1)
        self.params = [self.W, self.b]


    def neg_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x[T.arange(y.shape[0]), y]))

    def errors(self, y):
        return T.mean(T.neq(self.pred, y))


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation = T.tanh):
        self.input = input
        W = np.asarray(
            rng.uniform(
                low = -np.sqrt(6. / (n_in + n_out)),
                high = np.sqrt(6. / (n_in + n_out)),
                size = (n_in, n_out)
            ),
            dtype = theano.config.floatX
        )

        if(activation == T.nnet.sigmoid):
            W *= 4

        b = np.asarray(np.zeros(n_out, ), dtype = theano.config.floatX)
        self.W = theano.shared(value = W, name = 'w', borrow = True)
        self.b = theano.shared(value = b, name = 'b', borrow = True)

        line_output = T.dot(input, self.W)  + self.b
        self.output = (line_output if activation is None
                       else activation(line_output))
        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.input = input

        self.hidden_layer = HiddenLayer(
            rng, input,
            n_in, n_hidden
        )
        self.regression_layer = Regression(
            self.hidden_layer.output,
            n_hidden, n_out
        )

        self.L1 = (abs(self.hidden_layer.W).sum() +
                   abs(self.regression_layer.W).sum())

        self.L2_sqr = ((self.hidden_layer.W ** 2).sum() +
                       (self.regression_layer.W ** 2).sum())

        self.neg_log_likelihood = self.regression_layer.neg_log_likelihood
        self.errors = self.regression_layer.errors
        self.pred = self.regression_layer.pred

        self.param = self.hidden_layer.params + self.regression_layer.params


def trainMLP(learning_rate = 0.01, L1_reg = 0.0, L2_reg = 0.0001,
             n_epoch = 1000, dataset = "..\\mnist.pkl.gz", batch_size = 20,
             n_hidden = 500):
    data_sets = loadData(dataset)
    train_set_x, train_set_y = data_sets[0]
    valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]

    rng = np.random.RandomState(1234)

    x = T.matrix('x')
    y = T.ivector('y')
    index = T.iscalar('index')

    #实例化类，类中仍然是符号表示式子
    classifier = MLP(rng, x, 28 * 28, n_hidden, 10)
    cost = classifier.neg_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    error = classifier.errors(y)

    #定义误差值
    gparams = [T.grad(cost, param) for param in classifier.param]
    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(classifier.param, gparams)]


    #为训练模型，验证模型与测试模型定义可调函数
    train_mode = function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x : train_set_x[index * batch_size : (index + 1) * batch_size],
            y : train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    valid_model = function(
        inputs = [index],
        outputs = error,
        givens = {
            x : valid_set_x[index * batch_size : (index + 1) * batch_size],
            y : valid_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    test_model = function(
        inputs = [index],
        outputs = error,
        givens = {
            x : test_set_x[index * batch_size : (index + 1) * batch_size],
            y : test_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    # mini batch
    n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow = True).shape[0] // batch_size


    #早停控制模块写在这里
    patience = 10000
    patience_increase = 2
    improvement_rate = 0.995

    validation_frequency = min(n_train_batches,patience / 2)

    best_validation_value = np.inf
    test_score = 0.0

    beg_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epoch ) and (not done_looping):
        epoch += 1  #总的训练轮数，训练过整个数据集
        for mini_batch in range(n_train_batches):
            train_mode(mini_batch)  #训练集中单个块的训练

            #求出当前已经训练过的块的数量
            iter = (epoch - 1) * n_train_batches + mini_batch

            #检查是否到验证
            if((iter + 1) % validation_frequency == 0):   # 此时 iter + 1 的原因是避免第一个输入块进行验证
                #进行验证
                validation_res = [valid_model(i) for i in range(n_valid_batches)]
                validation_value = np.mean(validation_res)

                #输出提示信息
                print(
                    'valid - epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        mini_batch + 1,
                        n_train_batches,
                        validation_value * 100.
                    )
                )

                if validation_value < best_validation_value: #当前性能有所提升
                    if validation_value < (best_validation_value * improvement_rate): #当前性能提升较大
                       patience = max(patience,iter * patience_increase)

                    #更新当前最佳的测试结果，并输出提示信息
                    best_validation_value = validation_value

                    #计算测试结果
                    test_res = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_res)

                    print(
                        'test - epoch %i, minibatch %i/%i, test error %f %%' %
                        (
                            epoch,
                            mini_batch + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    #with open('MLP_best_model.pkl','wb') as f:
                    #    pickle.dump(classifier,f)
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    print(
        "train complete, validation_error is : %f %%, test_error is : %f %%"%
        ((1 - best_validation_value) * 100., (1 - test_score) * 100.)
    )

    print(
        "count epoch : %i s,  with %.2f epoches/sec" %
        (epoch, (1. * epoch) / (end_time - beg_time))
    )

def predict():
    classifier = pickle.load(open('best_model.pkl'))

    predict_model = theano.function([classifier.input],classifier.prediction)

    dataset = 'mnist.pkl.gz'
    datasets = loadData(dataset)
    test_set_x,test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = trainMLP(test_set_x[:10])
    print ("predicted values for the first 10 examples in test_set : ")
    print(predicted_values)


if __name__ == "__main__":
    trainMLP()













