#!/bin/bash -

# This runs inside the guest, as 'user', driven as a systemd service (TBA)

cd /home/user
source ./config/params

. env/bin/activate

## Old-style
#ipython --matplotlib=notebook

## New-style
jupyter notebook --ip=0.0.0.0 --port=$port_jupyter --no-browser --notebook-dir=$notebook_dir

## defaults:
# jupyter notebook --ip=0.0.0.0 --port=8080 --no-browser --notebook-dir=./notebooks
