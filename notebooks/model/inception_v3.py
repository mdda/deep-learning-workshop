# Inception-v3, model from the paper:
# "Rethinking the Inception Architecture for Computer Vision"
# http://arxiv.org/abs/1512.00567
# Original source:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
# License: http://www.apache.org/licenses/LICENSE-2.0

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/inception_v3.pkl

import lasagne

from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer, Pool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.normalization import batch_norm

# For image manipulation
import numpy as np
import scipy

def imagefile_to_np(image_file):
  im_uint8 = scipy.misc.imread(image_file)
  #print(np.shape(im_uint8))
  return resize_image(im_uint8)

def resize_image(image):
  return scipy.misc.imresize(image, (299, 299))


def preprocess_orig(im):
  # Expected input: RGB uint8 image
  # Input to network should be bc01, 299x299 pixels, scaled to [-1, 1].
  
  ## This is pretty heavy for a single function...
  import skimage.transform

  im = skimage.transform.resize(im, (299, 299), preserve_range=True)
  im = (im - 128) / 128.
  im = np.rollaxis(im, 2)[np.newaxis].astype('float32')

  return im

def preprocess(im_uint8):
  # Expected input: RGB uint8 image
  # Input to network should be bc01, 299x299 pixels, scaled to [-1, 1].

  im_resized = scipy.misc.imresize(im_uint8, (299, 299))
    
  im = np.asarray(im_resized).astype(np.float32)

  im = (im - 128) / 128.
  im = np.rollaxis(im, 2)[np.newaxis].astype('float32')
    
  #im = im.transpose( (2,0,1) )  # CxHxW
  #im = np.expand_dims(im, axis=0)

  return im


def bn_conv(input_layer, **kwargs):
  l = Conv2DLayer(input_layer, **kwargs)
  l = batch_norm(l, epsilon=0.001)
  return l


def inceptionA(input_layer, nfilt):
  # Corresponds to a modified version of figure 5 in the paper
  l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

  l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
  l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

  l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
  l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
  l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

  l4 = Pool2DLayer(input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
  l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

  return ConcatLayer([l1, l2, l3, l4])


def inceptionB(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionC(input_layer, nfilt):
    # Corresponds to figure 6 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
    l3 = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

    l4 = Pool2DLayer(input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionD(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    l1 = bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
    l2 = bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionE(input_layer, nfilt, pool_mode):
    # Corresponds to figure 7 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2a = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
    l2b = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3a = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
    l3b = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

    l4 = Pool2DLayer(input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])


def build_network():
  net = {}

  net['input'] = InputLayer((None, 3, 299, 299))
  net['conv']   = bn_conv(net['input'],    num_filters=32, filter_size=3, stride=2)
  net['conv_1'] = bn_conv(net['conv'],   num_filters=32, filter_size=3)
  net['conv_2'] = bn_conv(net['conv_1'], num_filters=64, filter_size=3, pad=1)
  net['pool']   = Pool2DLayer(net['conv_2'],   pool_size=3, stride=2, mode='max')

  net['conv_3'] = bn_conv(net['pool'],   num_filters=80, filter_size=1)

  net['conv_4'] = bn_conv(net['conv_3'], num_filters=192, filter_size=3)

  net['pool_1'] = Pool2DLayer(net['conv_4'], pool_size=3, stride=2, mode='max')
  
  net['mixed/join'] = inceptionA(
      net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
  net['mixed_1/join'] = inceptionA(
      net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

  net['mixed_2/join'] = inceptionA(
      net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

  net['mixed_3/join'] = inceptionB(
      net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

  net['mixed_4/join'] = inceptionC(
      net['mixed_3/join'],
      nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

  net['mixed_5/join'] = inceptionC(
      net['mixed_4/join'],
      nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

  net['mixed_6/join'] = inceptionC(
      net['mixed_5/join'],
      nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

  net['mixed_7/join'] = inceptionC(
      net['mixed_6/join'],
      nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

  net['mixed_8/join'] = inceptionD(
      net['mixed_7/join'],
      nfilt=((192, 320), (192, 192, 192, 192)))

  net['mixed_9/join'] = inceptionE(
      net['mixed_8/join'],
      nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
      pool_mode='average_exc_pad')

  net['mixed_10/join'] = inceptionE(
      net['mixed_9/join'],
      nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
      pool_mode='max')

  net['pool3'] = GlobalPoolLayer(net['mixed_10/join'])

  net['softmax'] = DenseLayer(net['pool3'], num_units=1008, nonlinearity=lasagne.nonlinearities.softmax)

  return net



if __name__ == "__main__":
  #import imagenet
  #classes = imagenet.get_synset()
  #print(classes[:10])
  
  # Load pretrained model
  model = build_network()
  output_layer = model['softmax']
  
  import pickle
  params = pickle.load(open('../data/inception3/inception_v3.pkl'))
  print("Saved model params.keys = ", params.keys())
  #print("  License : "+params['LICENSE'])   # Apache 2.0
  classes = params['synset words']
  lasagne.layers.set_all_param_values(output_layer, params['param values'])
  
  #exit()

  image_files = [
    '../images/grumpy-cat_224x224.jpg',
    '../images/sad-owl_224x224.jpg',
    '../images/cat-with-tongue_224x224.jpg',
    '../images/doge-wiki_224x224.jpg',
  ]

  #images = np.vstack( [  for f in image_files ] )

  # Test pretrained model
  #input_var = T.tensor4('inputs')
  #net['input'].set_input_var(input_var)
  #output_class = lasagne.layers.get_output(model, deterministic=True)
  #predict = theano.function([ input_var ], [ output_class ])
  #predict = theano.function(input_var, output_class)
  
  for i, f in enumerate(image_files):
    im = imagefile_to_np(f)
    prob = np.array(lasagne.layers.get_output(output_layer, preprocess(im), deterministic=True).eval())

    top5 = np.argsort(prob[0])[-1:-6:-1]    
    
    print("\n%s" % (image_files[i],))
    for n, label in enumerate(top5):
      print('  #{} : {}'.format(n+1, classes[label]))    
    
    #predictions = predict([ im ])
    #print("%s -> %3d = %s" % ((image_files[i] + ' '*40)[0:40], prediction, classes[prediction]))
