# _*_ coding:utf-8 _*_
import sys
import os
sys.path.append("..")
from MLP.mutiLayerPerceptrons import loadData
from dA.dA import scale_to_unit_interval
from dA.dA import tile_raster_images

import timeit
import theano
from theano import tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import PIL.Image as Image


class RBM(object):
    def __init__(
            self,
            input = None,
            n_visible = 784,
            n_hidden = 500,
            W = None,
            hbias = None,
            vbias = None,
            numpy_rng = None,
            theano_rng = None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if W is None:
            init_W = np.asarray(
                a = numpy_rng.uniform(
                    low = -4 * np.sqrt(6./(n_hidden + n_visible)),
                    high = 4 * np.sqrt(6./(n_hidden + n_visible)),
                    size = (n_visible, n_hidden)
                ),
                dtype = theano.config.floatX,
            )
            W = theano.shared(
                value = init_W,
                name = 'W',
                borrow = True
            )

        if hbias is None:
            hbias = theano.shared(
                value = np.zeros(shape=n_hidden,
                                 dtype = theano.config.floatX),
                name = 'hbias',
                borrow = True
            )

        if vbias is None:
            vbias = theano.shared(
                value = np.zeros(shape = n_visible,
                                 dtype = theano.config.floatX),
                name = 'vbias',
                borrow = True
            )
        self.input = input
        if input is None:
            self.input = T.matrix('input')
        self.theano_rng = theano_rng
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.params = [self.W, self.hbias, self.vbias ]

    # 计算 free_energy
    def freeEnergy(self, v_sample):
        # v_sample 是 N * n_visible 的变量
        vbias_terms= T.dot(v_sample, self.vbias) # 得到一个 N*1 的矩阵
        wx_b = T.dot(v_sample, self.W)
        hidden_terms = T.sum(T.log(1 + T.exp(wx_b)), axis = 1) #沿行相加，得到一个N*1 的矩阵
        return -hidden_terms - vbias_terms # 返回一个 N*1 的矩阵

    def propUp(self, vis):
        prev_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [prev_sigmoid_activation, T.nnet.sigmoid(prev_sigmoid_activation)]

    def sampleHGivenV(self, v0_sample):
        prev_sigmoid_h1, h1_sigmoid = self.propUp(v0_sample)
        h1_sample = self.theano_rng.binomial(size = h1_sigmoid.shape,
                                             n = 1,
                                             p = h1_sigmoid,
                                             dtype = theano.config.floatX)
        return [prev_sigmoid_h1, h1_sigmoid, h1_sample]

    def propDown(self, hid):
        prev_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [prev_sigmoid_activation, T.nnet.sigmoid(prev_sigmoid_activation)]

    def sampleVGivenH(self, h0_sample):
        prev_sigmoid_v1, v1_sigmoid = self.propDown(h0_sample)
        v1_sample = self.theano_rng.binomial(size  = v1_sigmoid.shape,
                                             n = 1,
                                             p = v1_sigmoid,
                                             dtype = theano.config.floatX)
        return [prev_sigmoid_v1, v1_sigmoid, v1_sample]

    def gibbsHvH(self, h0_sample):
        v1_prev_sigmoid, v1_sigmoid, v1_sample = self.sampleVGivenH(h0_sample)
        h1_prev_sigmoid, h1_sigmoid, h1_sample = self.sampleVGivenH(v1_sample)
        return [v1_prev_sigmoid, v1_sigmoid, v1_sample,
                h1_prev_sigmoid, v1_sigmoid, h1_sample]

    def gibbsVhV(self, v0_sample):
        h1_prev_sigmoid, h1_sigmoid, h1_sample = self.sampleHGivenV(v0_sample)
        v1_prev_sigmoid, v1_sigmoid, v1_sample = self.sampleVGivenH(h1_sample)
        return [h1_prev_sigmoid, h1_sigmoid, h1_sample,
                v1_prev_sigmoid, v1_sigmoid, v1_sample]

    def getCostUpdates(self, lr = 0.1, persistent = None, k = 1):
        h1_sigmoid_prev, h1_sigmoid, h1_sample = self.sampleHGivenV(self.input)
        if persistent is None:
            chain_start = h1_sample
        else:
            chain_start = persistent

        (
            [
            nv_prev_sigmoids,
            nv_sigmoids,
            nv_samples,
            nh_prev_sigmoids,
            nh_sigmoids,
            nh_samples
        ],
            updates
        ) = theano.scan(
            fn = self.gibbsHvH,
            outputs_info = [None, None, None, None, None, chain_start],
            n_steps = k,
        )

        chain_end = nv_samples[-1] #重构结果

        cost = T.mean(self.freeEnergy(self.input)) - \
               T.mean(self.freeEnergy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant = [chain_end])

        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr, theano.config.floatX)

        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.getPseudoLikeLiHoodCost(updates)
        else:
            monitoring_cost = self.getReconstructionCost(updates, nv_sigmoids[-1])

        return monitoring_cost

    def getReconstructionCost(self, updates, sigmoid_nv):
        cross_entropy = T.mean(T.sum(self.input * T.log(sigmoid_nv) + \
                                     (1 - self.input) * T.log(1 - sigmoid_nv), axis = 1))

    def  getPseudoLikeLiHoodCost(self, updates):
        bit_i_idx = theano.shared(value = 0, name = 'bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.freeEnergy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.freeEnergy(xi_flip)
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost

def testRbm(learning_rate = 0.1, train_epoch = 15,
            dataset = '../mnist.pkl.gz', batch_size = 20,
            n_chains = 20, n_samples = 10, output_folder = 'rbm_plots',
            n_hidden = 500):
    datasets = loadData(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size

    index = T.iscalar()
    x = T.matrix('x')
    rng = np.random.RandomState(1234)
    theano_rng = RandomStreams(rng.randint(2**30))

    persistent_chain = theano.shared(
                                    value = np.zeros(shape = (batch_size,n_hidden),
                                              dtype = theano.config.floatX),
                                    borrow = True
                                     )
    rbm = RBM(input = x,n_hidden = n_hidden,numpy_rng = rng, theano_rng = theano_rng)

    cost, updates = rbm.getCostUpdates(persistent= persistent_chain, k = 15)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    os.chdir(output_folder)


    train_rbm = theano.function([index],cost,updates = updates,
                                givens = {
                                    x : train_set_x[index * batch_size:\
                                        (index + 1) * batch_size]
                                },
                                name = 'train_rbm')

    plotting_time = 0.
    start_time = timeit.default_timer()

    for epoch in range(train_epoch):
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
        print ('training epoch %d, cost is %f'%epoch, np.mean(mean_cost))

        plotting_start =timeit.default_timer()
        image = Image.fromarray(
            obj = tile_raster_images(
                X = rbm.W.get_value(borrow = True).T,
                image_shape = (28,28),
                tile_shape = (10,10),
                tile_space = (1,1)
            )
        )

        image.save('filters_at_epoch_%i.png'%epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()
    print("train took %f minutes"%((end_time - start_time - plotting_time) / 60.))

   # 训练结束，用训练的结果生成一个新的数据集
    number_of_test_samples = test_set_x.get_value(borrow = True).shape[0]

    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        np.asarray(test_set_x.get_value(borrow = True)[test_idx:test_idx + n_chains])
    )

    plot_every = 1000
    ([
        presig_hids,
        hid_mfs,
        hid_samples,
        presig_vis,
        vis_mfs,
        vis_samples
    ],
    updates) = theano.scan(
        rbm.gibbsVhV,
        outputs_info = [None,None,None,None,None,persistent_vis_chain],
        n_steps = plot_every
    )

    updates.update({persistent_vis_chain:vis_samples[-1]})

    sample_fn = theano.function(
        [],
        [vis_mfs[-1],vis_samples[-1]],
        updates = updates
    )

    image_data = np.zeros((29 * n_samples + 1, 29 * n_chains + 1),
                          dtype = 'uint8')
    for idx in range(n_samples) :
        vis_mfs,vias_samples = sample_fn()
        image_data[29 * idx : 29 * idx + 28,:] = tile_raster_images(
            X = vis_mfs,
            image_shape = (28,28),
            tile_shape = (1,n_chains),
            tile_spaceing = (1,1)
        )

        image = Image.fromarray(image_data)
        image.save("samples.png")

if __name__ == "__main__":
    testRbm()







