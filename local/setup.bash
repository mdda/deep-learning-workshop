#!/bin/bash -

# This runs locally, to set up a dev machine so that it can work with Jupyter notebooks, etc

#set -e
#set -x

source ../config/params

virtualenv --system-site-packages ../env
. ../env/bin/activate

pip install --upgrade pip

pip install -r ./config/requirements.txt 

#logopath=./env/lib/python2.7/site-packages/notebook/static/base/images
#mv ${logopath}/logo.png ${logopath}/logo-orig_260x56.png
#cp ./notebooks/images/logo.png ${logopath}/

# Create these directories safely, just so we know they're there
mkdir -p $notebook_dir
mkdir -p $tensorflow_dir

#echo "OMP_NUM_THREADS=4" >> ~/.bashrc
#echo "export OMP_NUM_THREADS" >> ~/.bashrc

# If we get this far, everything installed/initialised successfully.
# This string is detected in the guest afterwards.
echo '=== LOCAL CONFIGURATION FINISHED OK ==='
