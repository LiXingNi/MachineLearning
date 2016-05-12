# _*_ coding:utf-8 _*_
from RBM import RBM
import theano
from theano import tensor as T
from theano import function
from MLP.mutiLayerPerceptrons import loadData
from MLP.mutiLayerPerceptrons import HiddenLayer
from MLP.mutiLayerPerceptrons import Regression
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import timeit
import pickle


class DBN(object):
    def __init__(self,
                 np_rng,
                 theano_rng = None,
                 n_ins = 784,
                 hidden_layer_size = [500, 500],
                 n_outs = 10
                 ):
        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2**30))

        self.n_hidden_layers = len(hidden_layer_size)
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in range(self.n_hidden_layers):
            if i == 0:
                i_in_size = n_ins
                i_input = self.x
            else:
                i_in_size = hidden_layer_size[i - 1]
                i_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng = np_rng,
                                        input = i_input,
                                        n_in = i_in_size,
                                        n_out = hidden_layer_size[i],
                                        activation = T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            rbm_layer = RBM(input = i_input,
                            numpy_rng = np_rng,
                            theano_rng = theano_rng,
                            n_visible = i_in_size,
                            n_hidden = hidden_layer_size[i],
                            W = sigmoid_layer.W,
                            hbias = sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        self.logistic_layer = Regression(input = self.sigmoid_layers[-1].output,
                                         n_in = hidden_layer_size[-1],
                                         n_out = n_outs)
        self.params.extend(self.logistic_layer.params)

        self.finetune_cost = self.logistic_layer.neg_log_likelihood(self.y)
        self.final_errors = self.logistic_layer.errors(self.y)

    def pretrainingFunction(self, data_set_x, batch_size, k):

        index = T.lscalar('index')
        learning_rate = T.scalar("lr")

        batch_begin = batch_size * index
        batch_end  = batch_begin + batch_size

        pretrain_fns = []

        for rbm in self.rbm_layers:
            cost, updates = rbm.getCostUpdates(learning_rate, k = k)
            fn = function(
                [index,theano.In(learning_rate,value = 0.1)],
                cost,
                updates = updates,
                givens = {
                    self.x : data_set_x[batch_begin:batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def buildFineTuneFunction(self,
                              datasets,
                              batch_size,
                              learning_rate):
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow = True).shape[0] // batch_size

        index = T.lscalar("index")

        gparams = T.grad(self.finetune_cost,self.params)

        updates = []
        for param, gparam in zip(self.params,gparams):
            updates.append((param,param - learning_rate * gparam))

        train_function = function(
            inputs = [index,],
            outputs = self.finetune_cost,
            updates = updates,
            givens = {
                self.x : train_set_x[batch_size * index : batch_size * (index + 1)],
                self.y : train_set_y[batch_size * index : batch_size * (index + 1)]
            }
        )

        valid_function_i = function(
            inputs = [index],
            outputs = self.final_errors,
            givens = {
                self.x : valid_set_x[batch_size * index: batch_size * (index + 1)],
                self.y : valid_set_y[batch_size * index: batch_size * (index + 1)]
            }
        )

        test_function_i = function(
            inputs = [index],
            outputs = self.final_errors,
            givens = {
                self.x : test_set_x[batch_size * index: batch_size * (index + 1)],
                self.y : test_set_y[batch_size * index: batch_size * (index + 1)]
            }
        )

        def valid_score():
            valid_res = []
            for i in range(n_valid_batches):
                valid_res.append(valid_function_i(i))
            return valid_res

        def test_score():
            test_res = []
            for i in range(n_test_batches):
                test_res.append(test_function_i(i))
            return test_res

        return train_function, valid_score, test_score


def testDBN(finetune_lr = 0.1,
            pretraining_epoch = 100,
            pretraining_lr = 0.01,
            k = 1,
            training_epoch = 1000,
            datasets = "../mnist.pkl.gz",
            batch_size = 10):

        dataset = loadData(datasets)
        train_set_x, train_set_y = dataset[0]
        valid_set_x, valid_set_y = dataset[1]
        test_set_x, test_set_y = dataset[2]

        n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow = True).shape[0] // batch_size


        np_rng = np.random.RandomState(123)
        dbn = DBN(
            np_rng = np_rng,
            n_ins= 28 *28,
            n_outs = 10
        )

        pretrain_functions = dbn.pretrainingFunction(train_set_x, batch_size, k)
        train_function, valid_score, test_score = dbn.buildFineTuneFunction(dataset, batch_size, finetune_lr)

        # 预训练
        start_time = timeit.default_timer()
        for layer in range(dbn.n_hidden_layers):  #逐层预训练
            for epoch in range(pretraining_epoch):
                cost = [pretrain_functions[layer](index = i, lr = pretraining_lr) \
                        for i in range(n_train_batches)]
                print "pre train %d for %d epoch,cost is %f" % (layer,epoch, np.mean(cost))

        end_time = timeit.default_timer()

        print "pre train spend time : %f minutes" % ((end_time - start_time) / 60.)

        with open("pretrain.pkl",'wb') as f:
            pickle.dump(dbn.params,f)

        print "-------------------- pretrain terminate ---------------------------"

        #使用早停微n_batches 调

        start_time = timeit.default_timer()
        patience = 4 * n_train_batches
        patience_increase = 2.
        improve_threshold = 0.995
        validation_frequency = min(n_train_batches, patience / 2)


        best_cost = np.inf
        test_cost= 0.
        epoch = 0
        done_looping = False

        while(epoch < training_epoch and not done_looping):
            for mini_batch in range(n_train_batches):
                curr_cost = train_function(mini_batch)
                iter = mini_batch + epoch * n_train_batches

                if((iter + 1) % validation_frequency == 0):
                    print "come in validation test-------"
                    valid_cost_lis = valid_score()
                    valid_cost = np.mean(valid_cost_lis)
                    if(valid_cost < best_cost):
                        print "curr cost down to %f at epoch %d/%d" % (curr_cost, epoch, mini_batch)
                        if(curr_cost <= best_cost * improve_threshold):
                            print "----------add patience"
                            patience = max(patience, iter * patience_increase)
                        best_cost = valid_cost

                        test_cost_lis = test_score()
                        test_cost = np.mean(test_cost_lis)
                        print "curr test socre is %f" % test_cost
                        with open("finetuen.pkl","wb") as f:
                            pickle.dump(dbn.params,f)


                if iter > patience:
                    done_looping = True
                    break
        end_time = timeit.default_timer()

        print "fine tune cost time %f minutes" % ((end_time - start_time)/60.)




if __name__== "__main__":
    testDBN()






















