#!/bin/bash -

# This runs locally, to set up a dev machine so that it can work with Jupyter notebooks, etc

source ./config/params
#echo ${port_jupyter}
echo ${notebook_dir}

## local hack
notebook_dir=${notebook_dir/\/home\/user\//\./} 
echo ${notebook_dir}
#exit 0

. ./env/bin/activate

## New-style
jupyter notebook --ip=0.0.0.0 --port=$port_jupyter --no-browser --notebook-dir=$notebook_dir

## defaults for theano using the CPU:
# export THEANO_FLAGS='mode=FAST_RUN,device=cpu,floatX=float32,blas.ldflags=-lopenblas'

## defaults for theano using the GPU:
# export THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32'
# ??export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64


#http://deeplearning.net/software/theano/library/config.html
# python -c 'import theano; print(theano.config)' | less


# rsync -avz -e 'ssh -p 8282' user@localhost:/home/user/notebooks/data/googlenet/*.pkl .
