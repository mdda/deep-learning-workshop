#!/bin/bash -

set -e
set -x

source ./config/params

# http://redcatlabs.com/downloads/deep-learning-workshop/LICENSE
RCL_BASE=http://redcatlabs.com/downloads/deep-learning-workshop/notebooks/data

rm -rf ./clean
mkdir -p ./clean

rsync -avz --progress \
     --exclude=".git/" --exclude=".geany/" \
     --exclude="__pycache__/" --exclude=".ipynb_checkpoints/" \
     --exclude="data/" --exclude="cache/" \
     --include="*/" --include="*.ipynb" \
     --exclude="*" \
     ./notebooks ./clean/
rm ./clean/notebooks/all-in-one.ipynb

rsync -avz --progress \
     ./notebooks/images ./clean/notebooks/

rsync -avz --progress \
     --exclude=".git/" \
     --exclude="__pycache__/" --exclude=".ipynb_checkpoints/" \
     --exclude="syntaxnet/" \
     ./notebooks/models ./clean/notebooks/

rsync -avz --progress \
     ./notebooks/images ./clean/notebooks/

# Temporary clean-out
rm ./clean/notebooks/2-CNN/8-Speech/SpeechAnalysis_*

# Link up the images directory
ln -s ./clean/images ./clean/notebooks/2-CNN/5-TransferLearning/

# Copy over the locally installed keras zoo model parameters
mkdir -p ./clean/dot_keras/models
rsync -avz --progress ~/.keras/models/nasnet_mobile.h5 ./clean/.keras/models/
rsync -avz --progress ~/.keras/models/nasnet_mobile_no_top.h5 ./clean/.keras/models/
