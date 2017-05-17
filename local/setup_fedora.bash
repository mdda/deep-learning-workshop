#!/bin/bash -

# This runs locally, to set up a dev machine so that it can work with Jupyter notebooks, etc

#set -e
#set -x

# First, let's check we've got the right RPMs installed (assumes Fedora)
rpms=`rpm -qa`
rpms_extra=''

#rpms_required="python python-devel python2-virtualenv python-pip"
#rpms_required="${rpms_required} python2-scipy python2-numpy python2-Cython"

rpms_required="python3 python3-devel python3-virtualenv python3-pip"
rpms_required="${rpms_required} python3-scipy python3-numpy python3-Cython"

rpms_required="${rpms_required} gcc gcc-c++ redhat-rpm-config"
rpms_required="${rpms_required} blas-devel lapack-devel atlas-devel"

FC_VER=`uname -r`
#if [ "${FC_VER/.fc23.}" != "${FC_VER}" ]; then
#  echo "Fedora 23"
#  rpms_required="${rpms_required} python-scikit-learn python-pandas python-pillow"
#else

echo "Fedora 25"
#rpms_required="${rpms_required} python2-scikit-learn python2-pandas python2-pillow"
rpms_required="${rpms_required} python3-scikit-learn python3-pandas python3-pillow"

#fi

### GPU installation:  
# For TensorFlow 1.1 : cudnn v5 supported.  v6+ are NOT!
# dnf install cuda-cudnn5.1-devel
### GPU installation

rpms_required="${rpms_required} graphviz"


for check in ${rpms_required}
do 
  found=`echo $rpms | grep $check`
  if [ -z "$found" ]; then
    #echo "Need to install $check (using 'dnf install $check') before running this setup"
    rpms_extra="$rpms_extra $check"
  fi
done

if [ -z "$rpms_extra" ]; then
  echo "Required RPMs are installed"
else
  echo "Additional RPMs are required before this setup can proceed."
  echo "###########################################################"
  echo "Please run (as 'root') : "
  echo "  dnf install$rpms_extra"
  echo "###########################################################"
  exit 1
fi

source ./config/params

#virtualenv --system-site-packages ./env2

virtualenv -p python3 --system-site-packages ./env3
. ./env3/bin/activate

pip install --upgrade pip

### GPU installation:
#pip install tensorflow-gpu
#python -c "import tensorflow as tf;sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))" | grep name
# or (maybe) :
#python -c "import tensorflow as tf;with tf.device('/gpu:0'):a=tf.constant([1.0]);tf.Session().run(a)"
### GPU installation

### GPU installation:  
#pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl
#python -c "import torch;print(torch.cuda.is_available())"
### GPU installation


pip install -r ./config/requirements.txt 

python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger

##logopath=./env/lib/python2.7/site-packages/notebook/static/base/images
#logopath=./env3/lib/python3.5/site-packages/notebook/static/base/images
#mv ${logopath}/logo.png ${logopath}/logo-orig_260x56.png
#cp ./notebooks/images/logo.png ${logopath}/

# Create these directories safely, just so we know they're there
notebook_dir=${notebook_dir/\/home\/user\//\./} 
mkdir -p $notebook_dir

tensorflow_dir=${tensorflow_dir/\/home\/user\//\./} 
mkdir -p $tensorflow_dir

#echo "OMP_NUM_THREADS=4" >> ~/.bashrc
#echo "export OMP_NUM_THREADS" >> ~/.bashrc

# If we get this far, everything installed/initialised successfully.
# This string is detected in the guest afterwards.
echo '=== LOCAL CONFIGURATION FINISHED OK ==='
