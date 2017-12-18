#### Ideas

*  Run-down of [recent advances in the literature](http://jiwonkim.org/awesome-rnn/)

*  Fancy RNN methods : [Attention and Augmented Recurrent Neural Networks](http://distill.pub/2016/augmented-rnns/)

*  [More reinforcement learning](http://www.wildml.com/2016/10/learning-reinforcement-learning/)
   *  https://github.com/dennybritz/reinforcement-learning
   
*  Re-check TensorFlow memory usage fror VGG16 / Inception3(or4), since TensorFlow seems to be in higher demand, frankly
   *  Just a moment, VGG seems like the largest model, but isn't the earliest nor the latest ... Timeline :
      *   AlexNet (2012 ImageNet = 15.4% Top5) became 
      *   ZFNet (2013 ImageNet = 14.8% Top5)
      *   GoogLeNet (2014 ImageNet = 6.67% Top5) == Inception-v1: 
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


### TensorFlow and Deep Learning Singapore - (2017-02-16) DONE

https://github.com/tensorflow/models/tree/master/slim

*  GoogLeNet = inception1 (2014)

*  inception3 (2015)
   *  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

*  inception4 (2016)
   *  wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz




### Next workshop venue : FOSSASIA - (2017-03-18 @16:55)  1hr

I gave a Deep Learning talk last year at FOSSASIA.  This was followed by more talks within the same subject at PyConSG and FifthElephant (India).

Since the last FOSSASIA, the Deep Learning Workshop repo (on mdda's GitHub) has been extended substantially.  
Depending on the time allotted, we'll be able to tackle 1 or 2 'cutting edge' topics in Deep Learning.  
Participants will be able to install the working examples on their own machines, and tweak and extend them for themselves.  

Like last year, the Virtual Box Appliances will be distributed on USB drives : The set-up has been proven to work well.  
Since this is hands-on, some familiarity with installing, running and playing with software will be assumed.  
Depending on demand, I can also do a quick intro about Deep Learning itself, 
though that would be pretty well-trodden ground that people who are interested would have seen several times before.

*   1hr <--- This is what they've asked for

This looks interesting ::
  https://aiexperiments.withgoogle.com/ai-duet
Also ::
  Drawing from edges (cats?)
  
Also :: seq2seq ?
  https://research.fb.com/downloads/babi/
  
  http://cs.mcgill.ca/~rlowe1/interspeech_2016_final.pdf
  https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
  wget https://people.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip

      
### TensorFlow and Deep Learning MeetUp talk - (2017-03-21)  30mins
  
Intro to CNNs : 

*  Re-work the FOSSASIA presentation to integrate more demo per minute...


See:
*  https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/


#### Do soon :

*   Description of difference between old and new style TensorFlow ```Evaluator()``` calling
*   CNN for MNIST
*   Adversarial images
*   Auto-encoders


### TensorFlow and Deep Learning MeetUp talk - (2017-04-13)  30mins

Focus is on GANs.  Essentially, 'my turn' to do advanced topic.
  
*  Would be cool to link in Tacomatic paper to Stamp speech thing somehow.   
*  Using Keras




### DONE : Implement googlenet in Keras for model zoo
  Good post, but requires new BN layer def, etc
    http://joelouismarino.github.io/blog_posts/blog_googlenet_keras.html

  Use Googlenet slim saved model into pure Keras version
    wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
    tar -xzf inception_v1_2016_08_28.tar.gz
    http://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file

  PR : https://github.com/fchollet/deep-learning-models/pull/59
  
  

### TODO : Implement DenseNet in Keras for model zoo
    https://github.com/liuzhuang13/DenseNet
  


Find papers for :
  Shape of 1-d profile from start to minimum and beyond
    _LIVE/Backprop/GoodFellow-2015_Optimisation-is-StraightLine_1412.6544v5.pdf 
    
  Eggcarton minima
    The Loss Surfaces of Multilayer Networks
    _INBOX/_LossSurfaceOfDeepNets_1412.0233.pdf 
      We show that for large-size decoupled networks the lowest critical values of the random loss function form 
      a layered structure and they are located in a well-defined band lower-bounded by the global minimum.
      
    Qualitatively characterizing neural network optimization problems  (Goodfellow+ 2014)
      https://arxiv.org/abs/1412.6544
    
  Information of Layers
    Opening the Black Box of Deep Neural Networks via Information
    _INBOX/_TwoPhasesOfSGD_1703.00810.pdf 
    _INBOX/_NN-InformationTheory-SGD_1703.00810.pdf 
  
  Rethinking generalisation
    Understanding Deep Learning Requires Re-Thinking Generalization
    _INBOX/_RethinkingGeneralisation_a667dbd533e9f018c023e21d1e3efd86cd61c365.pdf 
    Hmm : 
      https://www.reddit.com/r/MachineLearning/comments/5utu1p/d_a_paper_from_bengios_lab_contradicts_rethinking/
      https://openreview.net/pdf?id=rJv6ZgHYg

      An empirical analysis of the optimization of deep network loss surfaces
        https://arxiv.org/abs/1612.04010
      
      Qualitatively characterizing neural network optimization problems by Ian J. Goodfellow, Oriol Vinyals, Andrew M. Saxe
        https://arxiv.org/abs/1412.6544
    
      Flat Minima (1997)
        http://www.bioinf.jku.at/publications/older/3304.pdf
  
  Spatial pyramid pooling


## For PyTorch (2017-07-06) :
  DeepMind Relation-Networks ("RN"): 
    https://arxiv.org/abs/1706.01427
  PyTorch implementation :
    Implementation of "Sort-of-CLEVR"
    https://github.com/kimhc6028/relational-networks
    https://github.com/mdda/relational-networks
  bAbI : 
    Keras   : http://smerity.com/articles/2015/keras_qa.html
    PyTorch : https://github.com/thomlake/pytorch-notebooks/blob/master/mann-babi.ipynb
      


## TODO : Keras introductory example
  https://medium.com/towards-data-science/stupid-tensorflow-tricks-3a837194b7a0
    https://github.com/thoppe/tf_thomson_charges



  
## For PyTorch (2017-10-19) "TTS" ??
  Voice Synthesis for in-the-Wild Speakers via a Phonological Loop
    https://ytaigman.github.io/loop/
    Paper : https://arxiv.org/abs/1707.06588
    Code? : https://github.com/ytaigman/loop  (follow author to get an alert)
    Code  : https://github.com/facebookresearch/loop

    WORLD Vocoder
      Base : Masanori Morise, Fumiya Yokomori, and Kenji Ozawa. World: A vocoder-based high-quality speech synthesis system for real-time applications. IEICE TRANSACTIONS on Information and Systems, 99(7):1877–1884, 2016
        Paper   : https://www.jstage.jst.go.jp/article/transinf/E99.D/7/E99.D_2015EDP7457/_pdf
        Website : http://ml.cs.yamanashi.ac.jp/world/english/introductions.html
      D4c  : Masanori Morise. D4c, a band-aperiodicity estimator for high-quality speech synthesis. Speech Communication, 84:57–65, 2016.
        Paper: https://ecantorix.moe/synthesis/mbrola/mmorise_d4c.pdf
        Code : https://github.com/mmorise/World (BSD : See : http://ml.cs.yamanashi.ac.jp/world/english/faq.html)
        Code : https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
      TEST:  Does round-trip of parameters work?

    Merlin Toolkit
      Zhizheng Wu, Oliver Watts, and Simon King.  Merlin: An Open Source Neural Network Speech Synthesis System, pages 218–223. 9 2016.
        Paper   : http://ssw9.net/papers/ssw9_PS2-13_Wu.pdf
        Code    : https://github.com/CSTR-Edinburgh/merlin  (Apache 2.0)
        Project : http://www.cstr.ed.ac.uk/projects/merlin/
        Sample  : (output via ?WORLD) https://cstr-edinburgh.github.io/merlin/demo.html
        
    SampleRNN
      Website : http://www.josesotelo.com/speechsynthesis/  (though actual samples seem to be missing)
      Paper   : https://openreview.net/forum?id=B1VWyySKx
      Code    : https://github.com/soroushmehr/sampleRNN_ICLR2017  (MIT)  
      Code    : https://github.com/sotelo/parrot - RNN-based generative models for speech. 

    Graves attention model 
      Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013
        https://arxiv.org/abs/1308.0850  (43 pages)
          Handwriting example - has incremental window, defined on p26/corner26
          Effects of 'priming' for handwriting synthesis, shown p37/corner37 onwards
        
    
    
### Restructuring REPO
  Individual projects should go into their own repos to increase discoverability
  eg:
    BubbleBreaker                                  << mostly DONE
    CNN for Voice Stamps                           << mostly DONE
    Transfer Learning with SVM
    Keras GoogLeNet (including slim-to-keras)
    Captioning (including AIAYN)
    ChooseGPU
  and then pull in these repos suitably for the main VM creation
  Also : 
    Add RPM proxy mechanism
    /or/ think about doing it via docker and a wrapper...
  
  Now : 
    Migrate off Theano fully
    Contemplate installing PyTorch
    Or, instead, making use of tf.eager


## hinton_says_we_should_scrap_back_propagation
  https://www.reddit.com/r/MachineLearning/comments/70e4ex/n_hinton_says_we_should_scrap_back_propagation/

  [–]Optrode 136 points 2 days ago 

>So, I'm gonna offer a sort of outside perspective, 
which is the perspective of a neuroscience researcher who has only a basic understanding of ML. 
>
>I can see differences between how information is processed in the brain and in ANNs, 
but of course the caveat is that I have no clue which (if any) of those differences represent opportunities for improvement via biomimicry.
>
>That said, the notable differences I see between brains and deep learning models are:
>
>*    Sensory systems in the brain usually have a great deal of top down modulation 
(think early layers receiving recurrent input from later layers). 
There aren't really any sensory or motor systems in the brain that AREN'T recurrent.
>
>*    Sensory systems in the brain also tend to have a lot of lateral inhibition (i.e. neurons inhibiting other neurons in the same layer).
>
>*    Brain sensory systems tend to separate information into channels. 
e.g. at all levels of the visual system, there are separate pathways for high and low spatial frequency content 
(outline &amp; movement vs. texture), and color information.
>
>*    Particularly with regard to the visual system, inputs are always scanned in a dynamic fashion. 
When a person views a picture, only a very small subsection of the image 
(see: fovea, saccade) is seen at high detail at any instant. 
The "high detail zone" skips around the image, lingering on salient points.
>
>*    Obviously, there's STDP. 
STDP essentially pushes neurons to predict the future, 
and I think that unsupervised training methods that focus on predicting the future 
(this came up in the recent AMA, as I recall) obtain some of the same benefits as STDP.
>
>*    I've seen several comments in this thread on how reducing the number of weights per node (e.g. CNN, QRNN) is beneficial, 
and this resembles the state of affairs in the brain. 
There is no such thing as a fully connected layer in the brain, 
connectivity is usually sparse (though not random). 
This usually is related to the segregation of different channels of information.
>
>*    Lastly, most information processing / discrimination in the brain is assisted by semantic information. 
If you see a person in a hospital gown, you are primed to see a nurse or doctor. 
This remains true for a while afterwards, since we rarely use our sensory facilities to view collections of random, unrelated photos.
>
>
>    I read the wiki for STDP but didn't quite get a full understanding. Would you be able to talk a bit about it?

>Sure! It's actually pretty simple.
>
>Suppose we have two neurons, A and B. A synapses onto B ( A->B ). 
The STDP rule states that if A fires and B fires after a short delay, 
the synapse will be potentiated (i.e. B will increase the 'weight' assigned to inputs from A in the future).
>
>The magnitude of the weight increase is inversely proportional to the delay between A firing and B firing. 
So, if A fires and then B fires ten seconds later, the weight change will be essentially zero. 
But if A fires and B fires ten milliseconds later, the weight update will be more substantial.
>
>The reverse also applies. 
If B fires first, then A, then the synapse will weaken, and the size of the change is again inversely proportional to the delay.
>
>ELI5 version: STDP is a rule that encourages neurons to 'pay more attention' to inputs that predict excitation. 
Suppose you usually only bring an umbrella if you have reasons to think it will rain (weather report, you see rain outside, etc.). 
Then you notice that if you see your neighbor carrying an umbrella, 
even though you haven't seen any rain in the forecast, 
but sure enough, a few minutes later you see an updated forecast (or it starts raining). 
This happens a few times, and you get the idea: Your neighbor seems to be getting this information (whether it is going to rain) 
before your current sources. 
So in the future, you pay more attention to what your neighbor is doing.


      [–]cbeak 2 points 16 hours ago 

      I think when taking the brain as inspiration, the main question is which kinds of neural computations are necessary 
      and which are merely biological artifacts/spandrels. 
      
      Superficially, short-term plasticity strikes me as an artifact because it results from neurotransmitter depletion. 
      
      Spikes are necessary to avoid noise build-up, and depletion seems to be basically an artifact of this adaptation. 
      And even if depletion is evolved to be more pronounced and useful for computations such as gain-control 
      (down-regulating high-frequency inputs and up-regulating low-frequency input) and high- or low-band-filtering (which it appears to be), 
      it remains a question whether one would lose important computations if one leaves out such details. 
      
      I could image that each additional kind of computation will make a suitable learning rule more complicated 
      because each kind comes with its own set of hyper-parameters (e.g. gain, band specifications and kernel sizes), 
      each of which must probably be balanced in just the right way to avoid positive feedback-loops and catastrophic forgetting. 
      
      I could also imagine that several different priors expressed in the kernel sizes are necessary such that 
      different neurons can efficiently extract temporal information that is interesting in real-world data 
      (basically from the milliseconds to the seconds scale). 
      
      It generally seems like we need 
      (1) plenty of different kinds of computations, 
      (2) a connectome with a stochastic but a fairly simple connection scheme, 
      (3) a free energy minimizing learning rule where the energy is measured by sparse and delayed rewards and prediction errors. 
      
      Among those computations will likely be multiple kinds of non-linearities, 
      spatial and temporal clustering, modulation, normalization, lateral inhibition, 
      plenty of modulatory feedback connections. 
      
      The last step might be a massive hyper-parameter search by evolution of embodied agents in a resource-constrained sim. 
      That's what I would bet my money on.



###  More explicit description of comms

[–]deathofamorty 1 point 1 day ago 

How is it that neurons A and B know when each other fires? Is there a special type of synapse or something?

[–]Optrode 6 points 23 hours ago 

>Well, assuming that A synapses onto B but there is no reciprocal connection, A does not know when B fires. 
B, the post-synaptic neuron, knows A fired because it receives synaptic input from that synapses when A fires. 
Altering that synaptic weight is (in the most common cases) something that B does. 
A does not have to actively participate, beyond simply having fired at the appropriate time (which B detects).
>
>The exact mechanism for the synaptic potentiation is not clear...  
We know what some of the mechanisms in some cases are. 
There is a type of glutamate receptor, the NMDA receptor, that is well known for its role in long term synaptic potentiation (LTP). 
The NMDA receptor acts as a coincidence detector: 
it will only allow calcium ions into the postsynaptic neuron if a synaptic signal is received 
when the postsynaptic neuron is already depolarized to a positive voltage (i.e. activated).
>
>Mind you, that's extremely ELI5. 
There's a lot more to it, such as the fact that what actually matters is whether 
the DENDRITE (input structure of the neuron) is depolarized, not the whole cell, 
and those don't necessarily go hand in hand. 
Exactly how strongly the depolarization of the neuron's cell body depolarizes any particular dendrite branch will 
depend on the structure of the branch, and this can make it so that certain other synaptic inputs 
(a neuron has an average of 7000) may have a greater effect on whether synapses on a 
particular dendrite are in a state to be strengthened by LTP.
>
>Dendrites also have other cool properties, like how it's possible for a certain type of inhibitory input (Cl- channel mediated inhibition, as opposed to K+ channel mediated inhibition) to be capable of canceling out only certain excitatory inputs, but not others, as well as controlling how readily the neuron can be excited by repeated excitatory inputs (vs. requiring all the excitatory input to arrive at once).
>
>Which kind demonstrates another important difference between artificial neural networks and real neurons.. The "neurons" in an ANN are mostly linear, they just have a nonlinear activation function. Inputs are linearly summed. Real neurons do not linearly sun their inputs, the whole process of receiving input is nonlinear as fuck.



[–]timtom85 2 points 9 hours ago 

>I also checked out STDP after reading this and what caught my attention was the weakening of the weight if an input fires slightly after the neuron does. It seems very important to me because it can effectively get rid of spurious correlations, it can suppress feedback loops, and it can weed out unnecessary connections.




[–]CireNeikual 5 points 2 days ago 

What about TargetProp? It works without differentiable functions, it can be used with STDP/Hebbian learning (with appropriate discrete timesteps Hebbian and STDP can be equivalent).

I personally like revisting old methods and seeing how they fair with some new upgrades. Adaptive Resonance Theory, Self-Organizing Maps, or any other kind of vector quantizer. When in an appropriate architecture, they can do some interesting things. Interestingly, as soon as one abandons the need for differentiable functions and embraces sparsity, online/lifelong/incremental learning becomes much easier. This also leads to a performance boost, as one doesn't need many decorrelated replay samples in order to update. Further, with sparsity, sparse updates are possible, giving a further performance boost.

The human brain is quite sparse (it's the function of inhibitory neurons), so I feel like this is the right direction to take. Sparsity leads to low processing power use, something I feel this field desperately needs, with all the big projects taking fat GPU-filled server racks.

    permalinkembedsaveparentreportgive goldreply

[–]nobackprop 1 point 1 day ago 

I'll repeat here what I wrote elsewhere in this thread.

There is only one viable solution to unsupervised learning, the one used by the brain. It is based on spike timing. The cortex tries to find order in sensory discrete signals or spikes. The only type of order that can be found in spikes is temporal order. Here is the clincher: spikes are either concurrent or sequential. I and others have been saying this for years. Here's a link, if you are interested:

Why Deep Learning Is a Hindrance to Progress Toward True AI

It's all about timing.

    permalinkembedsaveparentreportgive goldreply

[–]mindbleach 0 points 1 day ago 

Train another network to guess future coefficients. So basically, still backprop at heart, but faster and more chaotic. Leap blindly downhill on gradient descent.

Early on, maybe keep the shitty random values, but change the connections.




## Papers
  -  PriorOut (distributional arguement as extension to DropOut)
     -  SignOut (discarded)
     -  Replace DropOut(==0) with (==N(0,1)), for instance

  -  ImageNet as an attention game
    -  (same as :) Relationships from a Stream
  
  -  Word-order game instead of translation for embedding-in-context a la Socher
       -  Or word-substitution
       -  Or phrase-deletion
     -  Sentence discriminator should use k=~3 CNN layers to construct 'extra' embedding with which to reason
       -  Then use same extra embedding to train dependency, NER or POS models like Socher
       -  Test environment? : http://allennlp.org/papers/AllenNLP_white_paper.pdf
     
  -  Word dependencies within sentences via 'plates'
     -  Defined on a lattice, or via a MTCS-like game tree
     -  Perhaps have k=~3 CNN layer(s) where channels denote not just 'chose me', but :
        -  chose left/right neighbour 
        -  my expected neighbourhood is +/- ...  (like SSD)
           -  No need to make the choice unique - channels/positions can support each other
  
  
  
  -  Embeddings on graphs (Meh?)
  
  -  Neural learning depends on dwelling on useful stuff...
    -  Think about STDP : Isn't it more obvious what's going on?
    
  -  Conversational agents as magician master chess players
    
  -  MNIST-oneshot
     -  NAS-like structure opt
     -  one-shot on MNST examples
     
  -  Speech/audio to phonemes / emotion / speaker-id at different time scales
  
