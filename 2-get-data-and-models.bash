#!/bin/bash -

set -e
set -x

source ./config/params

# http://redcatlabs.com/downloads/deep-learning-workshop/LICENSE
RCL_BASE=http://redcatlabs.com/downloads/deep-learning-workshop/notebooks/data


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
    #wget http://sydney.edu.au/engineering/it/~matty/Shakespeare/shakespeare.tar.gz
    wget http://www.cs.usyd.edu.au/~matty/Shakespeare/shakespeare.tar.gz
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
  if [ ! -e "gru_2layer_trained.pkl" ]; then
    # Fall-back source
    wget ${RCL_BASE}/RNN/gru_2layer_trained.pkl
  fi
  mv gru_2layer_trained.pkl gru_2layer_trained_claims.pkl
fi


if [ ! -e "india.names.1990-5.txt.gz" ]; then
  mkdir -p india-names
  cd india-names/
    # 50Mb
    if [ ! -e "ap-names.txt" ]; then
      if [ ! -e "ap-names.txt.gz" ]; then
        wget https://archive.org/download/india-names-dataset/ap-names.txt.gz
      fi
      if [ ! -e "ap-names.txt.gz" ]; then
        # Fall-back source
        wget ${RCL_BASE}/RNN/ap-names.txt.gz
      fi
      gunzip ap-names.txt.gz
    fi
    grep '^199[01234]' ap-names.txt | sort -k2,2nr | head -250000 > ../india.names.1990-5.txt
  cd ..

  gzip india.names.1990-5.txt  # 700Kb of text

  rm -rf india-names/
fi

if [ ! -e "ALL_1-vocab.txt.gz" ]; then
  # Retrieve ALL_1-vocab.txt.gz from fall-back data store (file was derived from 1-billion-corpus)...
  wget ${RCL_BASE}/RNN/ALL_1-vocab.txt.gz
  
  # Uploaded using...
  #rsync -avz --progress ALL_1-vocab.txt.gz user@example.com:~/deep-learning-workshop/notebooks/data/RNN/
  
  #gzip ALL_1-vocab.txt
fi

if [ ! -e "en.wikipedia.2010.100K.txt" ]; then
  # Retrieve wikipedia dump
  wget http://corpora2.informatik.uni-leipzig.de/downloads/eng_wikipedia_2010_100K-text.tar.gz
  if [ ! -e "eng_wikipedia_2010_100K-text.tar.gz" ]; then
    # Fall-back source of the same data
    wget ${RCL_BASE}/RNN/eng_wikipedia_2010_100K-text.tar.gz
  fi
  tar -xzf eng_wikipedia_2010_100K-text.tar.gz
  mv eng_wikipedia_2010_100K-sentences.txt en.wikipedia.2010.100K.txt
  rm eng_wikipedia*
fi

if [ ! -e "glove.first-100k.6B.50d.txt" ]; then
  # Retrieve Glove data, see : http://nlp.stanford.edu/projects/glove/
  wget http://nlp.stanford.edu/data/glove.6B.zip
  unzip glove.6B.zip
  head -100000 glove.6B.50d.txt > glove.first-100k.6B.50d.txt
  rm glove.6B.*
  
  if [ ! -e "glove.first-100k.6B.50d.txt" ]; then
    # Fall-back source of the same data
    wget ${RCL_BASE}/RNN/glove.first-100k.6B.50d.txt
  fi
fi

popd


# Game pre-trained weights

mkdir -p ./notebooks/data/game/crush
pushd ./notebooks/data/game/crush

# This is for Bubble Breaker
if [ ! -e "rl_10x14x5_2016-06-21_03-27.049999.pkl" ]; then
 wget ${RCL_BASE}/game/crush/rl_10x14x5_2016-06-21_03-27.049999.pkl
fi

popd


# Fall-back locations for the gloVe embedding
if [ '' ] && [ ! -e "glove.6B.300d.hkl" ]; then
  # Files in : ${RCL_BASE}/research/ICONIP-2016/
  #   507.206.240 Oct 25  2015 glove.6B.300d.hkl
  
  # Files in : ${RCL_BASE}/research/ICONIP-2016/  :: These are originals - citation desired...
  #    53.984.642 May 15 14:13 sparse.6B.300d_S-21_2n-shuf-noise-after-norm_.2.01_6-75_4000_GPU-sparse_matrix.hkl
  #   122.248.980 May  2 13:09 sparse.6B.300d_T-21_3500.1024@0.05-GPU-sparse_matrix.hkl
  #   447.610.336 May  2 13:04 sparse.6B.300d_T-21_3500.1024@0.05-GPU-sparsity_recreate.hkl
  #   160.569.440 May 14 14:57 vectors.2-17.hkl
fi

## Suitable 'styles' : picasso, etc


## Suitable 'base images' : people's faces, landscapes, etc


## Suitable 'commercial categories' : makes of car?

