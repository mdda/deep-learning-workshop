#!/bin/bash -

# This runs locally, to set up a dev machine so that it can work with Jupyter notebooks, etc

#set -e
#set -x

# First, let's check we've got the right packages installed (assumes Mac OSX Darwin)

pkgs_required="brew"
brew_pkgs_required="gcc python graphviz"
pip_pkgs_required="scipy numpy pandas bokeh matplotlib pillow graphviz sklearn jupyter theano lasagne nltk pydot pydot-ng glove_python"

FC_VER=`uname -s`
if [ "${FC_VER}" != "Darwin" ]; then
  echo "You are not running OSX Darwin. Please check your setup."
  exit 1
fi
echo "Your are using MAC OSX Darwin. Now we will proceed to install brew or upgrade brew" 
#start by installing brew if brew not present
which -s brew
if [[ $? != 0 ]] ; then
  echo "installing brew"
  ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
else
  echo "upgrading brew"
  brew update
  brew upgrade
fi

for check in ${brew_pkgs_required}
do 
  echo "installing $check using brew"
  brew install $check
done

echo "all pip runs will be made with --user flag"
echo "upgrading pip"
pip install --user --upgrade pip

for pkg in ${pip_pkgs_required}
do
  if [ "$pkg" != "glove_python" ]; then
    pip install --user --upgrade $pkg
  else
    echo "set export flag for CC=/usr/local/bin/gcc-6 from brew since glove_python does not work with clang"
    export CC=/usr/local/bin/gcc-6
    pip install --user --upgrade $pkg
  fi  
done

echo "install nltk packages"
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger

# If we get this far, everything installed/initialised successfully.
# This string is detected in the guest afterwards.
echo '=== LOCAL CONFIGURATION FINISHED OK ==='
