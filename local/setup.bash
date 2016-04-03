#!/bin/bash -

# This runs locally, to set up a dev machine so that it can work with Jupyter notebooks, etc

#set -e
#set -x

# First, let's check we've got the right RPMs installed (assumes Fedora)
rpms=`rpm -qa`
rpms_extra=''
for check in \
  python python-virtualenv \
  gcc gcc-c++ python-devel redhat-rpm-config \
  scipy numpy python-scikit-learn python-pandas Cython blas-devel lapack-devel atlas-devel python-pillow \
  graphviz
do 
  found=`echo $rpms | grep $check`
  if [ -z "$found" ]
  then
    #echo "Need to install $check (using 'dnf install $check') before running this setup"
    rpms_extra="$rpms_extra $check"
  fi
done

if [ -z "$rpms_extra" ]
then
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
