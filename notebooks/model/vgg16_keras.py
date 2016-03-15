from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import numpy as np

## Image manipulation :
#import cv2
#from PIL import Image
import scipy.misc


# From : https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

def full_classifier(weights_path=None):
  model = Sequential()
  model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation='relu'))
  
  #  Cut here to expose the 'pre-classification' features    
  
  model.add(Dropout(0.5))
  model.add(Dense(1000, activation='softmax'))

  if weights_path:
    print("  load_weights from '%s'" % (weights_path,))
    model.load_weights(weights_path)

  return model


def get_synset(path='../data/imagenet_synset_words.txt'):
  with open(path, 'r') as f:
    # Strip off the first word (until space, maxsplit=1), then synset is remainder
    return [ line.strip().split(' ', 1)[1] for line in f]

def imagefile_to_np(image_file):
  if True:
    image = scipy.misc.imresize(scipy.misc.imread(image_file), (224, 224))
    
    # Using PIL directly
    #image = Image.open(image_file, 'r')  # PIL reads as HxWxC  RGB
    #image = image.resize((224, 224), Image.ANTIALIAS)
    
    im_rgb = np.asarray(image).astype(np.float32)
    
    # http://stackoverflow.com/questions/4661557/pil-rotate-image-colors-bgr-rgb
    im = im_rgb[ ... , [2,1,0] ]

  if False:
    im = cv2.resize(cv2.imread(image_file), (224, 224)).astype(np.float32)    # OpenCV outputs them as BGR.
    
  im[:,:,0] -= 103.939 #B 
  im[:,:,1] -= 116.779 #G
  im[:,:,2] -= 123.68  #R
  
  im = im.transpose( (2,0,1) )  # CxHxW
  im = np.expand_dims(im, axis=0)

  return im


if __name__ == "__main__":
  classes = get_synset()
  #print(classes[:10])
  #exit()
  
  # Load pretrained model
  model = full_classifier('../data/VGG16/vgg16_weights.h5')
  #exit()

  image_files = [
    '../images/grumpy-cat_224x224.jpg',
    '../images/sad-owl_224x224.jpg',
    '../images/cat-with-tongue_224x224.jpg',
    '../images/doge-wiki_224x224.jpg',
  ]

  images = np.vstack( [ imagefile_to_np(f) for f in image_files ] )

  # Test pretrained model
  
  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy')
  
  #out = model.predict(images)
  #print( np.argmax(out) )
  
  for i, out in enumerate(model.predict_classes( images )):
    print("%s -> %3d = %s" % ((image_files[i] + ' '*40)[0:40], out, classes[out]))
