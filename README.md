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

* [DataScienceSG MeetUp](http://www.meetup.com/DataScience-SG-Singapore/) : 'Hardcore' session about Deep Learning (targeting 3 hours)
  *  Application : Classifying unknown classes of images (~transfer learning)
  *  Application-idea : Speech processing
  *  Application-idea : Character-wise name origins (LSTM/NLP application)


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
   *  'Commerce' - repurpose a trained network to classify our stuff
   *  'Art' - Style transfer with Lasagne, but using GoogLeNet features for speed
   *  'Reinforcement Learning' - learning to play "Bubble Breaker" 
   *  ?RNN - Updated with Shakespeare text, and bugfix

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
      *  First hour : up to end of ConvNetJS, and explain TheanoBasics
      *  Up to 1:30 : MNIST and ImageNet examples
      *  15 mins for each of : 
         *  'Commerce'
         *  'Art'
      *  30 mins for 'Reinforcement Learning'


#### Still Work-in-Progress 

*  Create sync-to-latest-workbooks script to update existing (taken-home) VMs

*  Create additional 'applications' modules (some ideas are given below)

*  Create tensorboard demo
   
   
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
         *  Optimising use of replay : [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
         *  Without replay (and good introduction) : [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
   *  Potential to make Javascript renderer of Bubble Breaker written in Python
      *  Host within Jupyter notebook (to display game-state, and potentially play interactively)
      *  Game mechanics driven by Python backend
         *  [Python to Javascript](http://blog.thedataincubator.com/2015/08/embedding-d3-in-an-ipython-notebook/)
         *  And Round-trip [Python ... Javascript](https://jakevdp.github.io/blog/2013/06/01/ipython-notebook-javascript-python-communication/)
      *  Interface similar (i.e. identical) to ALR or PLE
         *  Idea for 'longer term' : Add this as an OpenAI Gym environment
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
      *  Should consider what a 'minibatch' would look like
         *  Training of batches of samples looks like experience replay
         *  Selection of next move requires 'a bunch' of feed-forward evaluations - number unknown
            *  Find average # of moves available during a game
            *  Find average # of steps played during a game
      *  Simple rules to follow:
         *  Select next move at random from list of available areas, equally weighted
         *  Select next move at random from list of available areas, weighted by score (or simply cell-count)
      
*  Reinforcement Learning demos (Karpathy, mostly in Javascript)
   *  [```ConvNetJS```](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
   *  [```ReinforceJS```](http://cs.stanford.edu/people/karpathy/reinforcejs/)
   *  [```RecurrentJS```](http://cs.stanford.edu/people/karpathy/recurrentjs/) - contains character RNN demo (PG essays)
   *  [...more Karpathy goodness](http://karpathy.github.io/2016/05/31/rl/) - in Python/numpy
      *  With a ['100-lines' of Python gist](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) 
   

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
   *  How about learning to generate names?  e.g : Adversarial RNNs
      *   https://archive.org/details/india-names-dataset
          *   ```grep '^1994' ap-names.txt | sort -k2,2nr | head -50```
      *   More traditional corpus to vocabulary (with counts, pre-sorted) :
          *   ```andrewsm@holland:/mnt/data/home/andrewsm/OpenSource/Levy/1-billion-sgns/counts.words.vocab```
          *   ```rsync -avz andrewsm@holland:/mnt/data/home/andrewsm/OpenSource/billion-placeholder/data/2-glove/ALL_1-vocab.txt ./notebooks/data/RNN/```

      
   
*  Music example?
   *  [Google's Magenta](http://magenta.tensorflow.org/welcome-to-magenta)
   
*  Generative Adversarial Networks (GANs) ?
   *  [Original Paper - Goodfellow, 2014](http://arxiv.org/abs/1406.2661)
   *  [Upper bound](https://github.com/Newmu/dcgan_code) performance-wise (for inspiration?)
   *  [Overview](http://soumith.ch/eyescream/)
   *  [Karpathy illustrates instability issues](http://cs.stanford.edu/people/karpathy/gan/)
   *  On MNIST somehow?
      *  [Vectorizing MNIST](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/)
   *  Very helpful [Reddit](https://www.reddit.com/r/MachineLearning/comments/3yuwyj/tutorial_on_generative_adversarial_networks_in/) posting
      *  [Helpful hints](http://torch.ch/blog/2015/11/13/gan.html) on Torch version 
      *  [MNIST using GAN and Adversarial Autoencoder](https://github.com/igul222/mnist_generative) in Theano + Lasagne
   
   
*  Anomaly detection
   *  Very promising auto-encoder approach : [code for MNIST](https://github.com/h2oai/h2o-training-book/blob/master/hands-on_training/anomaly_detection.md)
   *  [video](http://www.mathtube.org/lecture/video/deep-learning-image-anomaly-detection)
   *  [Light on details : Slides 6+](http://www.slideshare.net/agibsonccc/anomaly-detection-in-deep-learning-62473913)
   


*  Image Captioning...
   *   https://github.com/mjhucla/TF-mRNN
       *  Accepts VGG or Inception3 image features

*  Res-Net
   *   Original (Microsoft Research, ImageNet end-2015) : https://github.com/KaimingHe/deep-residual-networks
   *   https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py


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
