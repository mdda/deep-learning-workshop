#!/bin/bash -

# This runs inside the guest at first boot.

set -e
set -x

cd /home/user
source ./config/params

virtualenv --system-site-packages env
. env/bin/activate

pip install --upgrade pip
#pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

pip install -r ./config/requirements.txt 

python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger

# Once this ~/env is built, we can copy it out to the host machine for 'caching'

# cp ./notebooks/images/logo.png ./env/lib/python2.7/site-packages/IPython/html/static/base/images/
# cp ./notebooks/images/logo.png ./env/lib/python2.7/site-packages/notebook/static/base/images/  # original : 260x56

logopath=./env/lib/python2.7/site-packages/notebook/static/base/images
mv ${logopath}/logo.png ${logopath}/logo-orig_260x56.png
cp ./notebooks/images/logo.png ${logopath}/

# Create these directories safely, just so we know they're there
mkdir -p $notebook_dir
mkdir -p $tensorflow_dir


echo "OMP_NUM_THREADS=4" >> ~/.bashrc
echo "export OMP_NUM_THREADS" >> ~/.bashrc

# If we get this far, everything installed/initialised successfully.
# This string is detected in the guest afterwards.
echo '=== CONFIGURATION FINISHED OK ==='
