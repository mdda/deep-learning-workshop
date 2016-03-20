#!/bin/bash -

# This runs locally, to set up a dev machine so that it can work with Jupyter notebooks, etc

source ./config/params

. env/bin/activate
tensorboard --host 0.0.0.0 --port $port_tensorboard --logdir $tensorflow_dir
