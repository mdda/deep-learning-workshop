#!/bin/bash -

# This runs inside the guest, as 'user', driven as a systemd service

cd /home/user
source ./config/params

. env3/bin/activate

jupyter notebook --ip=0.0.0.0 --port=$port_jupyter --no-browser --notebook-dir=$notebook_dir


## defaults for using the CPU (if you want to run locally, have a look in ./local/):
# export THEANO_FLAGS='mode=FAST_RUN,device=cpu,floatX=float32,blas.ldflags=-lopenblas'

## defaults for using the GPU (if you want to run locally, have a look in ./local/):
# export THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32'
# ??export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64


#http://deeplearning.net/software/theano/library/config.html
# python -c 'import theano; print(theano.config)' | less
