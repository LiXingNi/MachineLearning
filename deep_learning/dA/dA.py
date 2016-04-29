# _*_ coding:utf-8 _*_
import sys
sys.path.append("..")

import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import  RandomStreams
import numpy as np
import timeit
from MLP.mutiLayerPerceptrons import loadData
import PIL.Image as Image

def scale_to_unit_interval(ndar, eps=1e-8):
  """ Scales all values in the ndarray ndar to be between 0 and 1 """
  ndar = ndar.copy()
  ndar -= ndar.min()
  ndar *= 1.0 / (ndar.max() + eps)
  return ndar
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
  """
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.

  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).

  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.

  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image

  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)

  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats

  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not


  :returns: array suitable for viewing as an image.
  (See:`Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.

  """

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2

  # The expression below can be re-written in a more C style as
  # follows :
  #
  # out_shape = [0,0]
  # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
  #                tile_spacing[0]
  # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
  #                tile_spacing[1]
  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
      assert len(X) == 4
      # Create an output np ndarray to store the image
      if output_pixel_vals:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
      else:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

      #colors default to 0, alpha defaults to 1 (opaque)
      if output_pixel_vals:
          channel_defaults = [0, 0, 0, 255]
      else:
          channel_defaults = [0., 0., 0., 1.]

      for i in range(4):
          if X[i] is None:
              # if channel is None, fill it with zeros of the correct
              # dtype
              out_array[:, :, i] = np.zeros(out_shape,
                      dtype='uint8' if output_pixel_vals else out_array.dtype
                      ) + channel_defaults[i]
          else:
              # use a recurrent call to compute the channel and store it
              # in the output
              out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
      return out_array

  else:
      # if we are dealing with only one channel
      H, W = img_shape
      Hs, Ws = tile_spacing

      # generate a matrix to store the output
      out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


      for tile_row in range(tile_shape[0]):
          for tile_col in range(tile_shape[1]):
              if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                  if scale_rows_to_unit_interval:
                      # if we should scale values to be between 0 and 1
                      # do this by calling the `scale_to_unit_interval`
                      # function
                      this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                  else:
                      this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                  # add the slice to the corresponding position in the
                  # output array
                  out_array[
                      tile_row * (H+Hs): tile_row * (H + Hs) + H,
                      tile_col * (W+Ws): tile_col * (W + Ws) + W
                      ] \
                      = this_img * (255 if output_pixel_vals else 1)
      return out_array



class dA(object):
    def __init__(self, np_rng,
                 theano_rng = None,
                 input = None,
                 n_visible = 786,
                 n_hidden = 500,
                 W = None,
                 b_hid = None,
                 b_via = None
                 ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2**30))

        self.theano_rng = theano_rng

        if input is None:
            self.x = T.matrix("input")
        else:
            self.x = input

        if W is None:
            init_w = np.asarray(
                np_rng.uniform(
                    low = -4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high = 4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size = (n_visible, n_hidden)
                ),
                dtype = theano.config.floatX
            )
            W = theano.shared(value = init_w, name = 'w', borrow = True)

        if b_hid is None:
            b_hid = theano.shared(
                value = np.zeros(shape = (n_hidden), dtype = theano.config.floatX),
                name = 'b_hid',
                borrow = True
            )

        if b_via is None:
            b_via = theano.shared(
                value = np.zeros(shape = n_visible, dtype = theano.config.floatX),
                name = 'b_via',
                borrow = True
            )

        self.W = W
        self.W_prime = self.W.T
        self.b = b_hid
        self.b_prime = b_via

        self.params = [self.W, self.b, self.b_prime]


    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size = input.shape,
                                        n = 1,
                                        p = 1 - corruption_level,
                                        dtype = theano.config.floatX) * input
    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstruct_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_update(self, corruption_level, learning_rate):

        corruption_x = self.get_corrupted_input(self.x, corruption_level)
        hidden_y = self.get_hidden_values(corruption_x)
        reconstruct_z = self.get_reconstruct_input(hidden_y)

        L = -T.sum(self.x * T.log(reconstruct_z) + (1 - self.x) * T.log(1 - reconstruct_z),axis = 1)
        cost = T.mean(L)

        gparams = T.grad(cost,self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params,gparams)
        ]

        return (cost, updates)

def train_dA(learning_rate = 0.1,
             corruption_level = 0.3,
             training_epoch = 15,
             dataset = '../mnist.pkl.gz',
             batch_size = 20,
             output_folder = 'dA_plots'
             ):

    datasets = loadData(dataset)
    train_set_x,train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size


    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    index = T.iscalar('index')
    x = T.matrix('x')

    da = dA(
        np_rng = rng,
        theano_rng = theano_rng,
        input = x,
        n_visible = 28 * 28,
        n_hidden = 500
    )

    cost,updates = da.get_cost_update(
        corruption_level = corruption_level,
        learning_rate = learning_rate
    )

    train_da = theano.function(inputs = [index],
                               outputs = cost,
                               updates = updates,
                               givens = {
                                   x : train_set_x[ batch_size * index : batch_size * (index + 1) ]
                               })
    beg_time = timeit.default_timer()

    for epoch in range(training_epoch):
        c_cost = []
        for mini_batchs in range(n_train_batches):
            c_cost.append(train_da(mini_batchs))
        print("train epoch %d, cost %f " % (epoch, np.mean(c_cost)))

    end_time  = timeit.default_timer()
    train_time = end_time - beg_time
    image = Image.fromarray(
        tile_raster_images(da.W.get_value(borrow = True).T,
                           img_shape = (28,28),
                           tile_shape = (10,10),
                           tile_spacing = (1,1))
    )
    image.save(output_folder)
    print("train time : %f",train_time)


if __name__ == "__main__":
    print "new"
    i = 0
    while c_level <= 1:
        file_name = str(i) + '.jpg'
        i += 1
        train_dA(corruption_level = c_level,output_folder= file_name)
        c_level += 0.2








