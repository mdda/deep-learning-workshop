#!/bin/bash -

# This runs locally, to set up a dev machine so that it can work with Jupyter notebooks, etc

#set -e
#set -x

# First, let's check we've got the right RPMs installed (assumes Fedora)
rpms=`rpm -qa`
rpms_extra=''

rpms_required="python python2-virtualenv"
rpms_required="${rpms_required} gcc gcc-c++ python-devel redhat-rpm-config"
rpms_required="${rpms_required} scipy numpy Cython blas-devel lapack-devel atlas-devel"

FC_VER=`uname -r`
if [ "${FC_VER/.fc23.}" != "${FC_VER}" ]; then
  echo "Fedora 23"
  rpms_required="${rpms_required} python-scikit-learn python-pandas"
else
  echo "Fedora 24+"
  rpms_required="${rpms_required} python2-scikit-learn python2-pandas"
fi
rpms_required="${rpms_required} python-pillow graphviz"


for check in ${rpms_require}
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
  exit
fi

source ./config/params

virtualenv --system-site-packages ./env
. ./env/bin/activate

pip install --upgrade pip

pip install -r ./config/requirements.txt 

#logopath=./env/lib/python2.7/site-packages/notebook/static/base/images
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
