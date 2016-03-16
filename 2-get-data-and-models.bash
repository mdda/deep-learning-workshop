#!/bin/bash -

set -e
set -x

source ./config/params


## MNIST data
# ready for Keras



## ImageNet synset words

mkdir -p ./notebooks/data
pushd ./notebooks/data

## thousand lines with synset IDs and names 
if [ ! -e "imagenet_synset_words.txt" ]; then
  wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
  gunzip caffe_ilsvrc12.tar.gz 
  tar -xf caffe_ilsvrc12.tar synset_words.txt
  mv synset_words.txt imagenet_synset_words.txt
  rm caffe_ilsvrc12.tar
fi

popd


mkdir -p ./notebooks/data/VGG
pushd ./notebooks/data/VGG

## VGG16 model (converted from Caffee, importable into Keras)
#   https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# This is/was for imagenet-VGG16-keras (HUGE)
## [ '' ] &&  :: disables the if statement
if [ '' ] && [ ! -e "vgg16_weights.h5" ]; then
  # 553Mb
  echo "*** ALERT ***"
  echo "YOU NEED TO DOWNLOAD THE VGG weights (533Mb) manually from Google Drive"
  echo "and save it in './notebooks/data/VGG16/vgg16_weights.h5' yourself"
  echo "See link at : https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3"
fi

# This is for theano-lasagne-styletransfer
if [ ! -e "vgg19_normalized.pkl" ]; then
  # 80Mb
  wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl
fi

popd


## inception_v3 model (Google)

mkdir -p ./notebooks/data/inception3
pushd ./notebooks/data/inception3

if [ ! -e "inception_v3.pkl" ]; then
  # 95Mb
  wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/inception_v3.pkl
fi

popd




## Suitable 'styles' : picasso, etc


## Suitable 'base images' : people's faces, landscapes, etc


## Suitable 'commercial categories' : makes of car?

