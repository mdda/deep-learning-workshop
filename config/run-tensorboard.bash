#!/bin/bash -

# This runs inside the guest, as 'user', driven as a systemd service

cd /home/user
source ./config/params

. env3/bin/activate
tensorboard --host 0.0.0.0 --port $port_tensorboard --logdir $tensorflow_dir

