#!/bin/bash -

# This runs inside the guest at first boot.

set -e
set -x

cd /home/user
source configure-vm.conf


virtualenv --system-site-packages env
. env/bin/activate
pip install --upgrade pip
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

pip install -r requirements.txt 
# Once this ~/env is built, we can copy it out to the host machine for 'caching'

# Create this directories safely
mkdir -p $notebook_dir
mkdir -p $tensorflow_dir

# If we get this far, everything installed/initialised successfully.
# This string is detected in the guest afterwards.
echo '=== CONFIGURATION FINISHED OK ==='
