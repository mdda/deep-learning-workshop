#!/bin/bash -

# This runs inside the guest, as 'user', driven as a systemd service (TBA)

cd /home/user
source configure-vm.conf

. env/bin/activate
ipython --matplotlib=notebook
