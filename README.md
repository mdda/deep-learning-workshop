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

*  Run-down of [recent advances in the literature](http://jiwonkim.org/awesome-rnn/)

*  Include javascript-based demos in the VM (or on the thumb-drive direction) to avoid "no-WiFi" show-stopper

*  Add Google's Tensorflow/JS demo : http://playground.tensorflow.org/

*  Reinforcement Learning demo (Python)
   *  [```DeeR```](http://deer.readthedocs.io/en/master/index.html) - ```theano```-based
   *  [```AgentNet```](https://github.com/yandexdataschool/AgentNet) - ```theano + lasagne```-based
      *  Examples include Atari space-invaders using OpenAI Gym
      *  In iPython notebooks!
   *  Need to understand whether OpenAI gym, ALE, or PLE (PyGame Learning Environment) can be seen from within non-X container 
      *  ALE : Asteroids, Space Invaders, Ms Pacman, Pong, Demon Attack
         *  More [programmer-centric](http://yavar.naddaf.name/ale/) details
            *  And ```.bin``` [finding/installation](https://groups.google.com/forum/#!topic/arcade-learning-environment/WMCrtTZPE2A)
         *  Can run with pipes, and save images every **x** frames (?dynamically loaded into Jupyter?)
         *  [Native Python Interface](https://github.com/bbitmaster/ale_python_interface/wiki/Code-Tutorial)
      *  [CNN used as pre-processor](http://www.slideshare.net/johnstamford/atari-game-state-representation-using-convolutional-neural-networks) to get learning time within reasonable bounds
      *  [Blog posting about RL using Neon](http://www.nervanasys.com/deep-reinforcement-learning-with-neon/)
      *  [Asynchronous RL in Tensorflow + Keras + OpenAI's Gym](https://github.com/coreylynch/async-rl)
         *  Optimising use of replay : [Prioritized Experience Replay](http://arxiv.org/pdf/1511.05952v4.pdf)
         *  Without replay (and good introduction) : [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/pdf/1602.01783v1.pdf)
   *  Potential to make Javascript renderer of Bubble Breaker
      *  Host within Jupyter notebook (to display game-state, and potentially play interactively)
      *  Game mechanics driven by Python backend
         *  [Python to Javascript](http://blog.thedataincubator.com/2015/08/embedding-d3-in-an-ipython-notebook/)
         *  And Round-trip [Python ... Javascript](https://jakevdp.github.io/blog/2013/06/01/ipython-notebook-javascript-python-communication/)
      *  Interface similar (i.e. identical) to ALR or PLE
      *  Learn to play using one-step look-ahead and deep-learned value function for boards
         *  Possible to add Monte-Carlo depth search too
      *  Difficulty : How to deal with random additional columns 
         *  Would prefer to limit time-horizon of game 
            *  Perhaps have a 'grey column' added with fixed (high) value as a reward
         *  May need to customize reward function, since it is (in principle) unbounded
            *  Whereas what's important is the relative value of the different actions (rather than their individual accuracy)
      *  Optimisation : Game symmetry under permutation of the colours
         *  WOLOG, can assume colour in bottom right is colour '1'
            *  But colouring in remainder still gives us 3*2*1 choices
            *  So that 6x as many training examples available than without re-labelling
            *  Perhaps enumerate off colours in bottom-to-top, right-to-left order for definiteness
               *  Cuts down redundency in search space, but may open up 'strange holes' in knowledge
      *  Idea for 'longer term' : Add this as an OpenAI Gym environment
            
      
      
*  Reinforcement Learning demos (Karpathy, mostly in Javascript)
   *  [```ConvNetJS```](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
   *  [```ReinforceJS```](http://cs.stanford.edu/people/karpathy/reinforcejs/)
   *  [```RecurrentJS```](http://cs.stanford.edu/people/karpathy/recurrentjs/) - contains character RNN demo (PG essays)
   *  [...more Karpathy goodness](http://karpathy.github.io/2016/05/31/rl/) - in Python/numpy
   

*  Natural Language Processing
   *  Character-wise RNN that's already included is very slow to converge
   *  Probably limited by size of dictionary / corpus for learning
   *  Idea : How about a character-wise name recognition (by region)?
      *  Initial step would be a character-embedding (Char2Vec)
      *  Then an LSTM ending with a 1-hot region guess :
         *  { China, India, Thailand, Malaysia, Japanese, Philippines, "European" }
      *  This would (a) be useful for NER, (b) kind of fun, and (c) require far less data that a full langauge model
      *  Potential References : 
         *  http://research.microsoft.com/en-us/groups/speech/0100729.pdf
            *   N-grams give about 73% accuracy
            *   ~Syllables give about 74% accuracy
            *   Ensemble is about 78% accuracy
         *  http://lrec.elra.info/proceedings/lrec2010/pdf/763_Paper.pdf
            *   Higher N-gram scores possible (MaxEnt-categorisation)
            *   Our name corpus is built on top of the following two major sources: 
                1) the LDC bilingual person name list and 
                2) the "Person nach Staat" (Person according to state) category of Wikipedia, which contains person names written in English texts from different countries.
   
*  Music example?
   *  [Google's Magenta](http://magenta.tensorflow.org/welcome-to-magenta)
   
Ahh - but now I see that the PyCon has only scheduled 1h30 for the workshop, rather than the 3h00 they originally suggested.  
There may be no time to do other fancy stuff...


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
