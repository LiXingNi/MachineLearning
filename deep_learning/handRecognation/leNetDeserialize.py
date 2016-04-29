import sys
sys.path.append("..")
from CNN.convolutionNN import LeNetPoolLayer
from MLP.mutiLayerPerceptrons import MLP
import theano.tensor as T
import pickle
import numpy as np
from theano import function

class leNet(object):
    def __init__(self,model_path = "leNet.pkl",batch_size = 1):

        params = pickle.load(open(model_path))

        x = T.matrix('x')
        y = T.ivector('y')

        nkerns = [20,50]
        rng = np.random.RandomState(23455)
        init_input = x.reshape((batch_size,1,28,28))

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


        layer0.W.set_value(params[0].get_value(),borrow = True)
        layer0.b.set_value(params[1].get_value(),borrow = True)
        layer1.W.set_value(params[2].get_value(), borrow = True)
        layer1.b.set_value(params[3].get_value(), borrow = True)
        mlp_layer.hidden_layer.W.set_value(params[4].get_value(), borrow = True)
        mlp_layer.hidden_layer.b.set_value(params[5].get_value(), borrow = True)
        mlp_layer.regression_layer.W.set_value(params[6].get_value(), borrow = True)
        mlp_layer.regression_layer.b.set_value(params[7].get_value(), borrow = True)

        self.mlp_layer = mlp_layer
        self.predict = function([x],mlp_layer.pred)

    def prediction(self,test_image):
        return self.predict(test_image)

