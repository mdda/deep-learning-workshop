## Deep Learning Workshop

This repo includes all scripts required to build a VirtualBox 'Appliance' (an easy-to-install pre-configured VM) 
that can be used by Deep Learning Workshop participants.

This workshop consists of an introduction to deep learning (from single layer networks-in-the-browser, 
then using the VM/Jupyter setup to train networks using Theano/Lasagne, 
and finally use pretrained state-of-the-art networks, such as GoogLeNet, in various applications) :

* [FOSSASIA 2016](http://2016.fossasia.org/) : Deep Learning Workshop (2hrs)
  *  Application : Generative art (~style transfer)
  *  Application : Classifying unknown classes of images (~transfer learning)
  *  Slides for the talk are [here](http://redcatlabs.com/2016-03-19_FOSSASIA-Workshop/)

* [PyCon-SG 2016](https://pycon.sg/schedule/presentation/94/) : Deep Learning Workshop (1.5hrs)
  *  Unfortunately, due to 'demand' for speaker slots, PyCon has only scheduled 1h30 for the workshop, rather than the 3h00 they originally suggested...
  *  Application : Reinforcement Learning
  *  Slides for the talk are [here](http://redcatlabs.com/2016-06-23_PyConSG-Workshop/)

* [DataScienceSG MeetUp](http://www.meetup.com/DataScience-SG-Singapore/) : 'Hardcore' session about Deep Learning (targeting 2.5 hours)
  *  Application : Anomaly Detection (mis-shaped MNIST digits)
  *  Application : Classifying unknown classes of images (~transfer learning)

* [Fifth Elephant, India](https://fifthelephant.in/2016/) : Deep Learning Workshop (targeting 6 hours)
  *  Application : Anomaly Detection (mis-shaped MNIST digits)
  *  Application : Classifying unknown classes of images (~transfer learning)
  *  Application : Generative art (~style transfer)
  *  Application : Reinforcement Learning
  *  Application : RNN Fun


**NB : Ensure Conference Workshop announcement / blurb includes VirtualBox warning label**

The VM itself includes : 

* Jupyter (iPython's successor)
  * Running as a server available to the host machine's browser
* Data
  * MNIST training and test sets
  * Trained models from two of the 'big' ImageNet winners
  * Test Images for both recognition, 'e-commerce' and style-transfer modules
* Tool chain (mostly Python-oriented)
  * Theano / Lasagne
  * Keras / Tensorflow is included, but not really used
    * Because Tensorflow demands far more RAM than Theano, and we can't assume that the VM will be allocated more than 2Gb

And this repo can itself be run in 'local mode', using scripts in ```./local/``` to :

*  Set up the virtual environment correctly
*  Run ```jupyter``` with the right flags, paths etc


### Status : Workshop WORKS!

#### Currently working well

*  Scripts to create working Fedora 23 installation inside VM
   *  Has working ```virtualenv``` with ```Jupyter``` and ```TensorFlow / TensorBoard```
*  Script to transform the VM into a VirtualBox appliance
   *  Exposing ```Jupyter```, ```TensorBoard``` and ```ssh``` to host machine

*  Locally hosted Convnet.js for :
   *  Demonstration of gradient descent ('painting')

*  Locally hosted TensorFlow Playground  for :
   *  Visualising hidden layer, and effect of features, etc


*  Create workshop notebooks (Lasagne / Theano)
   *  Basics 
   *  MNIST
   *  MNIST CNN
   *  ImageNet : GoogLeNet
   *  ImageNet : Inception 3
   *  'Anomaly Detection' - identifying mis-shaped MNIST digits
   *  'Commerce' - repurpose a trained network to classify our stuff
   *  'Art' - Style transfer with Lasagne, but using GoogLeNet features for speed
   *  'Reinforcement Learning' - learning to play "Bubble Breaker" 
   *  'RNN-Fun' - Discriminative and Generative RNNs

*  Notebook Extras
   *  U - VM Upgrade tool
   *  X - BLAS configuration fiddle tool
   *  Z - GPU chooser (needs Python's ```BeautifulSoup```)

*  Create rsync-able image containing :
   *  VirtualBox appliance image
      +  including data sets and pre-trained models
   *  VirtualBox binaries for several likely platforms
   *  Write to thumb-drives for actual workshop
      *  and/or upload to DropBox

*  Workshop presentation materials


#### Still Work-in-Progress 

*  Create sync-to-latest-workbooks script to update existing (taken-home) VMs

*  Create additional 'applications' modules (see 'ideas.md')

*  Monitor TensorBoard - to see whether it reduces its memory footprint enough to switch from Theano...
   
   

### Notes

#### Running the environment locally

First, run the ```./local/setup.bash``` and follow the instructions (such as installing additional RPMs, for which you'll need ```root```):
```
./local/setup.bash 
```

Once that runs without complaint, the Jupyter notebook server can be run using :

```
./local/run-jupyter.bash 
```


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
