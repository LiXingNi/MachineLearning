# _*_ coding:utf-8 _*_
from __future__ import division
import theano
import numpy as np
from theano import function
from theano import tensor as T
import gzip
import cPickle as pickle
import timeit


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
        train_set, valid_set, test_set = pickle.load(f)

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
                           dataset="mnist.pkl.gz", batch_size = 500):

    #读取数据
    datasets = loadData(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #获取数据的大小
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print("building model...")

    index = T.lscalar('index')
    x = T.matrix('x')
    y = T.ivector('y')

    #实例化分类对象
    classifier = SoftMaxRegression(input = x, in_dim = 28 * 28, out_dim = 10)
    cost = classifier.neg_log_likelihood(y)
    error = classifier.errors(y)


    #定义各种函数
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

    train_model = function([index],
                           outputs = cost,
                           updates = updates,
                           givens = {
                               x : train_set_x[index * batch_size : (index + 1) * batch_size],
                               y : train_set_y[index * batch_size : (index + 1) * batch_size]
                           })

    #开始训练模型，训练过程中使用早停
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995  # 当损失函数的值小于等于原来的 improvement 倍，则增大 patience值

    validation_frequency = min(n_train_batches,patience / 2)  #最少要在达到patience跳出前，执行两次 valid_set 检测

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epoch) and (not done_looping):
        epoch += 1

        for minibatch_index in range(n_train_batches):
            #训练模型
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                #对验证样例进行zero_one_loss count
                validation_loss = [valid_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_loss)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss: #当前性能有提升
                    if this_validation_loss < best_validation_loss * improvement_threshold: #当前性能提升比较大
                        patience = max(patience,iter * patience_increase) #提升较大时增加patience

                    best_validation_loss = this_validation_loss
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        ('epoch %i, minibatch %i / %i, test error of best model %f %%') %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    with open('best_model.pkl','wb') as f:
                        pickle.dump(classifier,f)

            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )%
        (best_validation_loss * 100., test_score * 100.)
    )

    print ('the code run for %d epoch, with % epochs/sec'%
           (epoch, 1.* epoch / (end_time - start_time)))

def predict():
    classifier = pickle.load(open('best_model.pkl'))

    predict_model = theano.function([classifier.input],classifier.prediction)

    dataset = 'mnist.pkl.gz'
    datasets = loadData(dataset)
    test_set_x,test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("predicted values for the first 10 examples in test_set : ")
    print(predicted_values)

if __name__ == "__main__":
    sgd_optimization_mnist()












