#!/bin/bash -

# This runs inside the guest, as 'user', driven as a systemd service (TBA)

cd /home/user
source configure-vm.conf

. env/bin/activate

## Old-style
#ipython --matplotlib=notebook

## New-style
jupyter notebook --ip=0.0.0.0 --port=$port_jupyter --no-browser --notebook-dir=$notebook_dir
