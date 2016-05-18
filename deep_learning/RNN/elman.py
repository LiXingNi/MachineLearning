# _*_ coding:utf-8 _*_

import numpy as np
import theano
from theano import function
import theano.tensor as T

class ElmanRnn(object):
    def __init__(self, ne, nh, nc, de, cs):
        '''
        :param ne: 总的词汇数量
        :param nh: 隐层节点数量
        :param nc: 词的分类数量
        :param de: 每个词扩展的维度
        :param cs: window size 的大小

        '''

        self.emb = theano.shared(
            value = np.asarray(
                a = 0.2 * np.random.uniform(low = -1.0, high = 1.0, size = (ne + 1, de)),
                dtype = theano.config.floatX
             ),
            name= 'emb',
            borrow = True)
        self.Wx =theano.shared(
            value = np.asarray(
                a = 0.2 * np.random.uniform(low = -1.0, high = 1.0, size = (cs * de, nh)),
                dtype = theano.config.floatX
            ),
            name = "Wx",
            borrow = True
        )

        self.Wh = theano.shared(
            value = np.asarray(
                a = 0.2 * np.random.uniform(low = -1.0, high = 1.0, size = (nh, nh)),
                dtype = theano.config.floatX
            ),
            name = "Wh",
            borrow = True
        )

        self.W = theano.shared(
            value = np.asarray(
                a = 0.2 * np.random.uniform(low = -1.0, high = 1.0, size = (nh, nc)),
                dtype = theano.config.floatX
            ),
            name = "W",
            borrow = True
        )


        self.h0 = theano.shared(
            value = np.zeros(
                shape = nh,
                dtype = theano.config.floatX
            ),
            name = 'h0',
            borrow = True
        )

        self.hb = theano.shared(
            value = np.zeros(
                shape = nh,
                dtype = theano.config.floatX
            ),
            name = 'hb',
            borrow = True
        )

        self.h = theano.shared(
            value = np.zeros(
                shape = nc,
                dtype = theano.config.floatX
            ),
            name = 'h',
            borrow = True
        )

        self.params = [self.emb, self.Wx, self.Wh, self.W, self.h0, self.hb, self.h]

        #输入训练是 n 次组成的一个句子，为每个词生成一个滑动窗口。这里需要再将滑动窗口中每个字映射到对应的词向量
        idx = T.imatrix('idx')
        x = self.emb[idx].reshape((idx.shape[0], de * cs))
        y = T.iscalar('y')  #最后一个词的标签

        def recurrence(x_val, h_val):
            h_t = T.nnet.sigmoid(T.dot(x_val, self.Wx) + T.dot(h_val, self.Wh) + self.hb)
            s_t = T.nnet.sigmoid(T.dot(h_t, self.W) + self.h)
            return  h_t, s_t

        [h_t, s_t], _ = theano.scan(  # _ 代表 theano.scan 返回的updates
            fn = recurrence,
            sequences = x,
            outputs_info = [self.h0, None ],
            n_steps = x.shape[0]
        )

        p_given_x = s_t
        p_give_last_word = s_t[-1,:]

        y_prediction = T.argmax(p_given_x, axis = 1)

        y_precision = -T.log(p_give_last_word[y])
        gparams = T.grad(y_precision, self.params)
        lr = T.fscalar('lr')
        updates = [(param, param - lr * gparam) for param, gparam in zip(self.params, gparams)]

        self.train = function(
            inputs = [idx, y, lr],
            outputs = y_precision,
            updates = updates
        )
        self.classifier = function(
            inputs = [idx],
            outputs = y_prediction
        )

        self.normalize = theano.function( inputs = [],
             updates = {self.emb:\
             self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})



