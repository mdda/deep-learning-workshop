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


mkdir -p ./clean/dot_keras/models
rsync -avz --progress ~/.keras/models/nasnet_mobile_no_top.h5 ./clean/.keras/models/