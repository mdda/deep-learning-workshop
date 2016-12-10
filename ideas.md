#### Ideas

*  Run-down of [recent advances in the literature](http://jiwonkim.org/awesome-rnn/)

*  Fancy RNN methods : [Attention and Augmented Recurrent Neural Networks](http://distill.pub/2016/augmented-rnns/)

*  [More reinforcement learning](http://www.wildml.com/2016/10/learning-reinforcement-learning/)
   *  https://github.com/dennybritz/reinforcement-learning
   
*  Re-check TensorFlow memory usage fror VGG16 / Inception3(or4), since TensorFlow seems to be in higher demand, frankly
   *  Just a moment, VGG seems like the largest model, but isn't the earliest nor the latest ... Timeline :
      *   AlexNet (2012 ImageNet = 15.4% Top5) became 
      *   ZFNet (2013 ImageNet = 14.8% Top5)
      *   GoogLeNet (2014 ImageNet = 6.67% Top5): 
          [Blog posting](http://joelouismarino.github.io/blog_posts/blog_googlenet_keras.html), 
          [Code in Keras](https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14), 
          and uses [googlenet_weights.h5 in Model.zip](http://joelouismarino.github.io/blog_posts/googlenet.zip) - 50Mb
          *   VGG also appeared this year (7.3% Top5)
      *   ResNet (2015 ImageNet = 3.57% Top5)
      *   Google's Inception-v3 (unofficial ImageNet = 3.46% Top5)
      *   Chinese Ensembles (2016 ImageNet = 2.99%? Top5)
      
   *  [Explanation of history of CNN models since LeNet](https://culurciello.github.io/tech/2016/06/04/nets.html)
   *  [TF-Slim model zoo](https://github.com/tensorflow/models/tree/master/slim)
      *  [Code for many generations of CNN models](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/nets)
      *  [VGG16 model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) - 490MB download
      *  [VGG16 code](https://github.com/tensorflow/models/blob/master/slim/nets/vgg.py)
      *  [VGG19 model](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) - 509MB download, 549MB checkpoint
   
*  [TensorFlow Resources](https://github.com/jtoy/awesome-tensorflow#)
   *  [Various DNNs in TensorFlow](https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32) using TF-slim
   *  and the [corresponding repository](https://github.com/awjuliani/TF-Tutorials) of code, including :
      *  DCGAN - An implementation of Deep Convolutional Generative Adversarial Network.
      *  InfoGAN An implementation of InfoGAN: Interpretable representation learning by information maximizing generative adversarial nets
      *  Deep Layer Visualization - Tutorial on visualizing intermediate layer activation during MNIST classification.
      *  Deep Network Comparison - Implementations of ResNet, HighwayNet, and DenseNet, for CIFAR10 classification.
      *  RNN-TF - Tutorial on implementing basic RNN in tensorflow.
      *  t-SNE Tutorial - Tutorial on using t-SNE to visualize intermediate layer representation during MNIST classification task.

*  Deciding on TensorFlow sugar :
   *  [CNN MNIST Rosetta stone](http://blog.mdda.net/ai/2016/11/26/layers-on-top-of-tensorflow)
   
*  [WaveNet: A Generative Model for Raw Audio](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
   *  Except that (according to Keras implementation), for 1 second of audio, using a downsized model (4000hz vs 16000 sampling rate, 16 filters v/s 256, 2 stacks vs ??):
      *  A Tesla K80 needs around ~4 minutes.
      *  A recent macbook pro needs around ~15 minutes. 
   *  Deepmind has reported that generating one second of audio with their model takes about 90 minutes.

*  [Neural Photo Editor](https://github.com/ajbrock/Neural-Photo-Editor)  
   *  Updated to work through a notebook?  - except that that's mainly a Javascript exercise, with little DL content

*  Investigate the real-ness of :
   *  [PCANet](https://arxiv.org/abs/1404.3606v2); and 
   *  [SARM](https://arxiv.org/abs/1608.04062v1) :: Ohh, there's already a retraction : (https://arxiv.org/pdf/1608.04062v2.pdf)
      *  In all, while it is all possible to construct a SARM-VGG model in hours, by choosing all subsets randomly, the performance will not be guaranteed. 
      *  The current implementation for SARM-VGG will bring in dramatically higher complexity and can take multiple days.
      *  Reddit comment : 
         *   Yes, but the part about sparse coding being the fixed point of that particular recurrent neural network defined in terms of the dictionary matrix provides a theoretical motivation for using K-SVD to learn the weights even in the "aggressive approximation".
         *   I found that part of the paper interesting. The confusing part was that in the main experiment on ImageNet they did not seem to use sparse coding at all, they instead seemed to use convolutional PCA or LDA, although that part was difficult to parse.

*  [Cocktail party problem](https://indico.io/blog/biased-debrief-of-the-boston-deep-learning-conference/)
  *  Signal processing problem where multiple speech signals are mixed in a single channel, 
     and the challenge is to separate the individual components (i.e., speakers) from the mix. 
  *  John Hershey from Mitsubishi Electric Research Labs talked through their solution using embedding vectors, then played samples that sounded really good! 
  *  Speech is just one of many kinds of noisy sequence, and it could be fun to explore other signal separation problems using a similar method. 

*  Andrew McCallum: structured knowledge graphs + neural networks
  *  Prof. McCallum was instrumental in developing conditional random fields. 
  *  He talked about a universal schema using structured knowledge bases, a neat take on helping models exploit "what is known" about the world to make better predictions. 
  *  He also talked about traversing graph structures as a sequence, and feeding that to sequence models like LSTM/recurrent neural networks—a known tactic that probably doesn’t get enough attention given the amount of knowledge locked in knowledge graphs.

*  Image Completion
   *  [http://bamos.github.io/2016/08/09/deep-completion/](On TensorFlow)
   *  https://news.ycombinator.com/item?id=12260853

*  Natural Language Processing
   *  Character-wise RNN that's already included is very slow to converge
   *  Probably limited by size of dictionary / corpus for learning
   
   *  Idea : Dual RNNs for NER (which I know 'cold')
      *   219554  818268 3281528 /home/andrewsm/SEER/external/CoNLL2003/ner/eng.train  == pretty small CoNLL training set
          *   BUT : The text for CoNLL-2003 isn't open source (needs Reuters agreement) - so that is out
          
      *   https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/models/word2vec_sample.zip  50Mb
      
      *   Instead : Let's install spacy (and shout out to honnibal) and a 'bunch of text' for it to annotate
          *   Except : that's 500Mb of data...  
          *   Perhaps just use its POS (in NLTK) : 
              *   ```pip install nltk  # from nltk.tag.perceptron import PerceptronTagger```
      *   And an [English language](http://corpora2.informatik.uni-leipzig.de/download.html) : 
          *   ```wget http://corpora2.informatik.uni-leipzig.de/downloads/eng_wikipedia_2007_100K-text.tar.gz```
          *   ```wget http://corpora2.informatik.uni-leipzig.de/downloads/eng_wikipedia_2010_100K-text.tar.gz```
          *   ```wget http://corpora2.informatik.uni-leipzig.de/downloads/eng_news_2015_100K.tar.gz```
      *   ```nltk.tokenize.punkt```

      *   Main thing to learn : NER (i.e. NNP) for single-case text


   *  Idea : How about a character-wise name recognition (by region)?
      *  Initial step would be a character-embedding (?Char2Vec)
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
          
   *  How about learning to generate English dictionary words?  e.g : Adversarial RNNs
      *   More traditional corpus to vocabulary (with counts, pre-sorted) :
          *   ```andrewsm@holland:/mnt/data/home/andrewsm/OpenSource/Levy/1-billion-sgns/counts.words.vocab```
          *   ```rsync -avz andrewsm@holland:/mnt/data/home/andrewsm/OpenSource/billion-placeholder/data/2-glove/ALL_1-vocab.txt ./notebooks/data/RNN/```
             *   ```wc ALL_1-vocab.txt  # 511438 1022864 6223250 ALL_1-vocab.txt```
          *   ```/usr/share/dict/linux.words``` for an alphabetical dictionary


   
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
         *  https://github.com/igul222/mnist_generative/blob/master/gan.py
         *  https://github.com/igul222/swft/tree/master/swft
   
   *  https://gist.github.com/rezoo/4e005611aaa4dad26697
   *  https://www.reddit.com/r/MachineLearning/comments/3yuwyj/tutorial_on_generative_adversarial_networks_in/
      *  http://dpkingma.com/sgvb_mnist_demo/demo.html
   *  http://www.kdnuggets.com/2016/07/mnist-generative-adversarial-model-keras.html
      *  https://github.com/osh/KerasGAN
      *  https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb
   
   
*  Anomaly detection II
   *  Something more comprehensive than the MNIST+Auto-encoder example

*  Image Captioning...
   *   https://github.com/mjhucla/TF-mRNN
       *  Accepts VGG or Inception3 image features

*  Res-Net
   *   Original (Microsoft Research, ImageNet end-2015) : https://github.com/KaimingHe/deep-residual-networks
   *   https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py


*  To what extent should this remain Theano/Lasagne?
   *   Main thrust away from TensorFlow was size of Keras/VGG model
   *   Have a look at TFSlim : 
       *   Unfortunately, its RNN 'story' is "we're looking into it", which is doubly disappointing
   *   Alternatively, just code some new notebooks in Tensorflow so that workshop can be 'dual'
   *   Look at Kadenze(?) course on ~'TensorFlow with style'

-------------------
# NEXT VENUES...

*  Ideas : http://www.kdnuggets.com/meetings/

-------------------
# DONE

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
   
*  Anomaly detection
   *  Very promising auto-encoder approach : [code for MNIST](https://github.com/h2oai/h2o-training-book/blob/master/hands-on_training/anomaly_detection.md)
   *  [video](http://www.mathtube.org/lecture/video/deep-learning-image-anomaly-detection)
   *  [Light on details : Slides 6+](http://www.slideshare.net/agibsonccc/anomaly-detection-in-deep-learning-62473913)


