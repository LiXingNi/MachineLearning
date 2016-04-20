import theano
from theano import function
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy as np
import pylab
from PIL import Image

rng = np.random.RandomState(23409805)

inputs = T.tensor4(name = "inputs")

w_shape = (3,3,9,9)
w_bound = np.sqrt(3*9*9)
W = theano.shared(
    np.asarray(rng.uniform(
        low = -1.0 / w_bound,
        high = 1.0 / w_bound,
        size = w_shape),
        dtype = inputs.dtype),
        name = "W"
    )

b_shape = (3,)
b=theano.shared(
    value = np.asarray(
        rng.uniform(
            low = -0.5,
            high = 0.5,
            size = b_shape),
        dtype = inputs.dtype),
    name = 'b'
)

conv_out = conv2d(inputs,W)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
f = function([inputs],output)

img = Image.open(open('1.jpg','rb'))
img = np.asarray(img,dtype = 'float32') / 256
img_ = img.transpose(2,0,1).reshape(1,3,img.shape[0],img.shape[1])
f_img = f(img_)

pylab.subplot(1,4,1); pylab.axis('off'); pylab.imshow(img)

pylab.gray()

pylab.subplot(1,4,2); pylab.axis('off');pylab.imshow(f_img[0,0,:,:])
pylab.subplot(1,4,3); pylab.axis('off');pylab.imshow(f_img[0,1,:,:])
pylab.subplot(1,4,4); pylab.axis('off');pylab.imshow(f_img[0,2,:,:])
pylab.show()



