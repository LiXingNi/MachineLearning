# _*_ coding:utf-8 _*_
from __future__ import division
from mutiLayerPerceptrons import *
import theano
from theano import tensor as T
from theano import function
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import numpy as np
import timeit


class LeNetPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, pool_size = (2, 2)):
        '''
        :param rng: np.random
        :param input: image
        :param filter_shape: output_node * input_feature_map_nums * filter_height * filter_width
        :param image_shape: batch_size * input_feature_map_nums * image_height * image_width
        :param pool_size:
        '''
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])) // np.prod(pool_size)
        w_bound = np.sqrt(6./(fan_in + fan_out))

        self.W = theano.shared(
            value = np.asarray(
                rng.uniform(low = -w_bound, high = w_bound, size = filter_shape),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        self.b = theano.shared(
            np.zeros((filter_shape[0], ),
                     dtype = theano.config.floatX),
            borrow = True
        )

        convo_out = conv2d(
            input = input,
            filters=self.W,
            input_shape = image_shape,
            filter_shape = filter_shape)

        pool_out = downsample.max_pool_2d(input = convo_out , ds = pool_size, ignore_border = True)

        self.output = T.nnet.relu(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.param = [self.W, self.b]




def trainLeNet(learning_rate = 0.1,
                n_epoch = 200,
                dataset = "mnist.pkl.gz",
                nkerns = [20, 50],
                batch_size = 500, test_image = None):

    data_sets = loadData(dataset)
    train_set_x, train_set_y = data_sets[0]
    valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]
    # mini batch
    n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow = True).shape[0] // batch_size

    rng = np.random.RandomState(23455)
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.iscalar('index')
    init_input =  x.reshape((batch_size, 1, 28, 28))

    layer0 = LeNetPoolLayer(
        rng,
        input=init_input,
        filter_shape = (nkerns[0], 1, 5, 5),
        image_shape = (batch_size, 1, 28, 28)
        )


    layer1 = LeNetPoolLayer(
        rng,
        input = layer0.output,
        filter_shape = (nkerns[1], nkerns[0], 5, 5),
        image_shape = (batch_size, nkerns[0], 12, 12)
    )


    mlp_layer_input = layer1.output.flatten(2)
    mlp_layer = MLP(rng, mlp_layer_input, nkerns[1] * 4 * 4, 500, 10)

    cost = mlp_layer.neg_log_likelihood(y)
    errors = mlp_layer.errors(y)

    params = layer0.param + layer1.param + mlp_layer.params

    gparams = [T.grad(cost,param) for param in params]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]

    train_model = function(
        [index],
        cost,
        updates = updates,
        givens = {
            x : train_set_x[index * batch_size : (index + 1) * batch_size],
            y : train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    valid_model = function(
        [index],
        errors,
        givens = {
            x : valid_set_x[index * batch_size : (index + 1) * batch_size],
            y : valid_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )
    test_model = function(
        [index],
        errors,
        givens = {
            x : test_set_x[index * batch_size : (index + 1) * batch_size],
            y : test_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    #早停开始
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    valid_frequency = min(n_train_batches,patience // 2)

    epoch = 0
    loop_done = False

    best_valid_score = np.Inf
    test_score = 0.0

    start_time = timeit.default_timer()


    #早停
    while (epoch <= n_epoch) and (not loop_done):
        epoch += 1

        for mini_batch in range(n_train_batches):
            train_model(mini_batch)

            iter = (epoch - 1) * n_train_batches + mini_batch

            #到达需要测试的地方
            if (iter + 1) % valid_frequency == 0:
                valid_result = [valid_model(index) for index in range(n_valid_batches)]
                valid_score = np.mean(valid_result)

                #输出本次测试结果
                print(
                    'valid - epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        mini_batch + 1,
                        n_train_batches,
                        valid_score * 100.
                    )
                )

                # valid值有提升
                if(valid_score < best_valid_score):
                #检测是否达到patience更新条件
                    if(valid_score <= best_valid_score * improvement_threshold):
                        patience = max(patience,iter * patience_increase)

                    best_valid_score = valid_score

                    #在检测性能提升的情况下测试 测试数据集
                    test_result = [test_model(index) for index in range(n_test_batches)]
                    test_score =  np.mean(test_result)

                    print(
                        'test - epoch %i, minibatch %i/%i, test error %f %%' %
                        (
                            epoch,
                            mini_batch + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    if test_image is not None:
                        test_result = []
                        for image in test_image:
                            test_result.append(mlp_layer.prediction_func(image))
                        print ("one test result is : ",test_result)

            if patience <= iter:
                loop_done = True
                break


    end_time = timeit.default_timer()

    print(
        "train complete, validation_error is : %f %%, test_error is : %f %%"%
        ((1 - best_valid_score) * 100., (1 - test_score) * 100.)
    )

    print(
        "count epoch : %i s,  with %.2f epoches/sec" %
        (epoch, (1. * epoch) / (end_time - start_time))
    )
    return mlp_layer


def leNetPrediction(image_set):
    classifier = trainLeNet()

    predict_model = theano.function([classifier.input],classifier.pred)

    for image in image_set:
        print(predict_model(image))


#if __name__ == "__main__":
#    trainLeNet()













