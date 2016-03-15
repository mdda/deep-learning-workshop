#!/bin/bash -

set -e
set -x

source ./vm-config/params


## MNIST data
# ready for Keras



## VGG16 model (converted from Caffee, importable into Keras)
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

mkdir -p ./notebooks/data/VGG16
pushd ./notebooks/data/VGG16

## thousand lines with synset IDs and names 
if [ ! -e "synset_words.txt" ]; then
  wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
  gunzip caffe_ilsvrc12.tar.gz 
  tar -xf caffe_ilsvrc12.tar synset_words.txt
  rm caffe_ilsvrc12.tar
fi

# The model itself
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
if [ ! -e "vgg16_weights.h5" ]; then
  echo "*** ALERT ***"
  echo "YOU NEED TO DOWNLOAD THE VGG weights manually from Google Drive"
  echo "and save it in './notebooks/data/VGG16/vgg16_weights.h5' yourself"
  echo "See link at : https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3"
fi

popd


## Suitable 'styles' : picasso, etc


## Suitable 'base images' : people's faces, landscapes, etc


## Suitable 'commercial categories' : makes of car?

