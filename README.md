## [FOSSASIA 2016](http://2016.fossasia.org/) : Deep Learning Workshop

This repo includes all scripts required to build the VirtualBox 'Appliance' (an easy-to-install pre-configured VM) 
to be used by the Deep Learning Workshop participants.

Slides for the talk are [here](http://redcatlabs.com/2016-03-19_FOSSASIA-Workshop/), and 
will probably be updated after the event too, since participants who want to play with the 
VirtualBox Appliance on their laptops when they get home might find a few more hints helpful.

The VM itself includes : 

* Jupyter (iPython's successor)
  * Running as a server available to the host machine's browser
* Data
  * MNIST training and test sets
  * Trained models from two of the 'big' ImageNet winners
  * Data directory will live inside image (heavy reliance on people being able to do VirtualBox)
* Tool chain (mostly Python-oriented)
  * Theano / Lasagne
  * Keras / Tensorflow is included, but not really used
    * Because Tensorflow demands far more RAM than Theano, and we can't assume that the VM will be allocated more than 2Gb

And this repo can itself be run in 'local mode', using scripts in ```./local/``` to :

*  Set up the virtual environment correctly
*  Run ```jupyter``` with the right flags, paths etc


### Status : Workshop WORKED!

#### Currently working well

*  Scripts to create working Fedora 23 installation inside VM
   *  Has working ```virtualenv``` with ```Jupyter``` and ```TensorFlow / TensorBoard```
*  Script to transform the VM into a VirtualBox appliance
   *  Exposing ```Jupyter```, ```TensorBoard``` and ```ssh``` to host machine

*  Tested out Convnet.js for :
   *  Demonstration of gradient descent
   *  Demonstration of MNIST learning

*  Ensure Conference Workshop announcement / blurb includes VirtualBox warning label

*  Create workshop notebooks (Lasagne / Theano)
   *  Basics 
   *  MNIST
   *  MNIST CNN
   *  ImageNet : GoogLeNet
   *  ImageNet : Inception 3
   *  'Commerce' - repurpose a trained network to classify our stuff
   *  'Art' - Style transfer with Lasange, but using GoogLeNet features for speed
   *  RNN - Updated with Shakespeare text, and bugfix

*  Notebook Extras
   *  X - BLAS configuration fiddle tool
   *  Z - GPU chooser (needs Python's ```BeautifulSoup```)

*  Create rsync-able image containing :
   *  VirtualBox appliance image
      +  including data sets and pre-trained models
   *  VirtualBox binaries for several likely platforms
   *  Write to thumb-drives for actual workshop

*  Workshop presentation materials
   *  Fairly short, to give time to play with the VMs
   *  Timings:
      *  First hour : up to end of ConvNetJS, and explain ThenoBasics
      *  Up to 1:30 : MNIST and ImageNet examples
      *  15 mins each : 'Commerce' and 'Art-Style-Transfer'


#### Still Work-in-Progress 

*  Create workshop notebooks
   *  Art
      +  Deep dream
      
*  Create sync-to-latest-workbooks script to update existing (taken-home) VMs

*  Create tensorboard demo

*  Workshop presentation materials
   *  Enhance, so that workshop module final results are given as pictures
      *  Because lots of people just want to listen, not do
   
#### Ideas

*  Reinforcement Learning demo
   *  [```DeeR```](http://deer.readthedocs.io/en/master/index.html) - ```theano```-based
   *  [```AgentNet```](https://github.com/yandexdataschool/AgentNet) - ```theano + lasagne```-based
      *  Examples include Atari space-invaders using OpenAI Gym
      *  In iPython notebooks!



### Notes

#### Running the environment locally




#### Git-friendly iPython Notebooks

Using the code from : http://pascalbugnion.net/blog/ipython-notebooks-and-git.html (and
https://gist.github.com/pbugnion/ea2797393033b54674af ), 
you can enable this kind of feature just on one repository, 
rather than installing it globally, as follows...

Within the repository, run : 
```
# Set the permissions for execution :
chmod 754 ./bin/ipynb_optional_output_filter.py

git config filter.dropoutput_ipynb.smudge cat
git config filter.dropoutput_ipynb.clean ./bin/ipynb_optional_output_filter.py
```
this will add suitable entries to ``./.git/config``.

or, alternatively, create the entries manually by ensuring that your ``.git/config`` includes the lines :
```
[filter "dropoutput_ipynb"]
	smudge = cat
	clean = ./bin/ipynb_output_filter.py
```

Note also that this repo includes a ``<REPO>/.gitattributes`` file containing the following:
```
*.ipynb    filter=dropoutput_ipynb
```

Doing this causes git to run ``ipynb_optional_output_filter.py`` in the ``REPO/bin`` directory, 
which only uses ``import json`` to parse the notebook files (and so can be executed as a plain script).  

To disable the output-cleansing feature in a notebook (to disable the cleansing on a per-notebook basis), 
simply add to its metadata (Edit-Metadata) as a first-level entry (``true`` is the default): 

```
  "git" : { "suppress_outputs" : false },
```


### Useful resources

* [MathJax](http://nbviewer.ipython.org/github/olgabot/ipython/blob/master/examples/Notebook/Typesetting%20Math%20Using%20MathJax.ipynb)
* [Bokeh](http://bokeh.pydata.org/en/latest/docs/quickstart.html)
