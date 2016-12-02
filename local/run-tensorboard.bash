#!/bin/bash -

# This runs locally, to set up a dev machine so that it can work with Jupyter notebooks, etc

source ./config/params
#echo ${port_tensorboard}
echo ${tensorflow_dir}

## local hack
tensorflow_dir=${tensorflow_dir/\/home\/user\//\./} 
echo ${tensorflow_dir}
#exit 0

. ./env3/bin/activate
tensorboard --host 0.0.0.0 --port $port_tensorboard --logdir $tensorflow_dir
