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


## GoogLeNet model (Google)

mkdir -p ./notebooks/data/googlenet
pushd ./notebooks/data/googlenet

if [ ! -e "blvc_googlenet.pkl" ]; then
  # 27Mb
  wget -N https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl
fi

popd



# MNIST dataset

mkdir -p ./notebooks/data/MNIST
pushd ./notebooks/data/MNIST

if [ ! -e "mnist.pkl.gz" ]; then
  # 15Mb
  wget http://deeplearning.net/data/mnist/mnist.pkl.gz
fi

popd



# RNN corpus and pre-trained weights

mkdir -p ./notebooks/data/RNN
pushd ./notebooks/data/RNN

if [ ! -e "claims.txt.gz" ]; then
  # 7.3Mb
  wget 'https://github.com/ebenolson/pydata2015/raw/master/4%20-%20Recurrent%20Networks/claims.txt.gz'
fi

if [ ! -e "Shakespeare.poetry.txt.gz" ]; then
  mkdir Shakespeare
  cd Shakespeare/
    # 20Mb
    wget http://sydney.edu.au/engineering/it/~matty/Shakespeare/shakespeare.tar.gz
    tar -xzf shakespeare.tar.gz 
  cd ..

  # NB: There are other folders here, but these are the simplest example sets
  cat Shakespeare/comedies/* Shakespeare/histories/* Shakespeare/tragedies/* > Shakespeare.plays.txt
  cat Shakespeare/poetry/* > Shakespeare.poetry.txt
  
  gzip Shakespeare.plays.txt   # 5Mb of text 
  gzip Shakespeare.poetry.txt  # 700Kb of text

  rm -rf Shakespeare/
fi

if [ ! -e "gru_2layer_trained_claims.pkl" ]; then
  # 1.7Mb
  wget 'https://github.com/ebenolson/pydata2015/raw/master/4%20-%20Recurrent%20Networks/gru_2layer_trained.pkl'
  mv gru_2layer_trained.pkl gru_2layer_trained_claims.pkl
fi


if [ ! -e "india.names.1990-5.txt.gz" ]; then
  mkdir -p india-names
  cd india-names/
    # 20Mb
    if [ ! -e "ap-names.txt" ]; then
      if [ ! -e "ap-names.txt.gz" ]; then
        wget https://archive.org/download/india-names-dataset/ap-names.txt.gz
      fi
      gunzip ap-names.txt.gz
    fi
    grep '^199[01234]' ap-names.txt | sort -k2,2nr | head -250000 > ../india.names.1990-5.txt
  cd ..

  gzip india.names.1990-5.txt  # 700Kb of text

  rm -rf india-names/
fi

if [ ! -e "ALL_1-vocab.txt.gz" ]; then
 # Retrieve ALL_1-vocab.txt from somewhere (1-billion-corpus)...
 gzip ALL_1-vocab.txt
fi

if [ ! -e "en.wikipedia.2010.100K.txt" ]; then
  # Retrieve wikipedia dump
  wget http://corpora2.informatik.uni-leipzig.de/downloads/eng_wikipedia_2010_100K-text.tar.gz
  tar -xzf eng_wikipedia_2010_100K-text.tar.gz
  mv eng_wikipedia_2010_100K-sentences.txt en.wikipedia.2010.100K.txt
  rm eng_wikipedia*
fi


popd







## Suitable 'styles' : picasso, etc


## Suitable 'base images' : people's faces, landscapes, etc


## Suitable 'commercial categories' : makes of car?

