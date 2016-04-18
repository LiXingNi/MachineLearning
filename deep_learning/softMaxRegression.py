import theano
import numpy as np
from theano import function
from theano import tensor as T
import gzip
import cPickle as pickle


class SoftMaxRegression(object):
    def __init__(self, input, in_dim, out_dim):
        self.input = input
        self.W = theano.shared(
            value = np.zeros((in_dim, out_dim), dtype=theano.config.floatX),
            name = 'W',
            borrow = True
        )

        self.b = theano.shared(
            value = np.zeros((out_dim, ), dtype = theano.config.floatX),
            name = 'b',
            borrow = True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_predict = T.argmax(self.p_y_given_x, axis = 1)
        self.param = [self.W, self.b]

    def neg_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x[T.arange(y.shape[0]), y]))

    def errors(self, y):
        #下面这项判断为检验判断，其实是为了引入y_predict的计算
        if y.ndim != self.y_predict.ndim:
            print("predict dim not equal")
        return T.mean(T.neq(self.y_predict, y))

def loadData(dataset):
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding = 'latin1')

    #定义函数内函数，用来将数据集传入GPU
    def shared_data(data_xy, borrow = True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype = theano.config.floatX), borrow = borrow)

        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_data(train_set)
    valid_set_x, valid_set_y = shared_data(valid_set)
    test_set_x, test_set_y = shared_data(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

#训练控制函数
def sgd_optimization_mnist(learning_rate=0.13, n_epoch = 1000,
                           dataset="mnist.pkl.gz", batch_size = 600):

    #读取数据
    datasets = loadData(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #获取数据的大小
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    print("building model...")

    index = T.scalar('index')
    x = T.matrix('x')
    y = T.vector('y')

    #实例化分类对象
    classifier = SoftMaxRegression(input = x, in_dim = 28 * 28, out_dim = 10)
    cost = classifier.neg_log_likelihood(y)
    error = classifier.errors(y)

    test_model = function([index], outputs = error,
                          givens = {
                              x: test_set_x[index * batch_size : (index + 1) * batch_size],
                              y: test_set_y[index * batch_size : (index + 1) * batch_size]
                          })
    valid_model = function([index], outputs = error,
                           givens = {
                               x : valid_set_x[index * batch_size : (index + 1) * batch_size],
                               y : valid_set_y[index * batch_size : (index + 1) * batch_size]
                           })
    g_W = T.grad(cost,classifier.W)
    g_b = T.grad(cost,classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = function([index],outputs = cost,)








