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

# Link up the images directories
ln -s ../../images ./clean/notebooks/2-CNN/5-TransferLearning/

rsync -avz --progress \
     ./presentation ./clean/

# Link the presentation image directory into the tree too (already there)
#ln -s ../../presentation/reveal.js-2.6.2/img ./clean/notebooks/images/presentation


# Temporary clean-out
rm ./clean/notebooks/2-CNN/8-Speech/SpeechAnalysis_*


# Copy over the locally installed keras zoo model parameters
mkdir -p ./clean/.keras/models
rsync -avz --progress ~/.keras/models/nasnet_mobile.h5 ./clean/.keras/models/
rsync -avz --progress ~/.keras/models/nasnet_mobile_no_top.h5 ./clean/.keras/models/
