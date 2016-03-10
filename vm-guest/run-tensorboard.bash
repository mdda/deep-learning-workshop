#!/bin/bash -

# This runs inside the guest, as 'user', driven as a systemd service (TBA)

cd /home/user
source configure-vm.conf

. env/bin/activate
tensorboard --host 0.0.0.0 --port $port_tensorboard --logdir /home/user/tensorflow.logdir
