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

Or speech recognition? : 
  https://github.com/llSourcell/tensorflow_speech_recognition_demo
    https://www.youtube.com/watch?v=u9FPqkuoEJ8

    TIMIT 
      http://www.cstr.ed.ac.uk/research/projects/artic/mocha.html
      http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3
      https://catalog.ldc.upenn.edu/ldc93s1
      http://en.pudn.com/downloads329/doc/detail1449116_en.html
  
  https://github.com/buriburisuri/speech-to-text-wavenet
    36,395 sentences in the VCTK corpus with a length of more than 5 seconds to prevent CTC loss errors. 
    VCTK corpus can be downloaded from http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz. 
    After downloading, extract the 'VCTK-Corpus.tar.gz' file to the 'asset/data/' directory.
      10Gb of WAVs : DOWNLOADING
      
  https://svail.github.io/mandarin/

  http://www.visbamu.in/viswaDataset.html
  
  http://www.speech.cs.cmu.edu/comp.speech/Section1/Data/ldc.html
  
  2006 NIST Spoken Term Detection Evaluation Set
  
  ASpIRE Audio 
  
  http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/8kHz_16bit/
  
  http://forcedalignment.blogspot.sg/2015/06/building-acoustic-models-using-kaldi.html
  
  Standard ASR Test Sets   Size 
    Wall Street Journal 80 hrs 
    TED-LIUM 118 hrs 
      http://www.openslr.org/7/
    Switchboard 300 hrs 
    Libri Speech 960 hrs 
    Fisher English 1800 hrs 
    ASpIRE (Fisher Training) 1800 hrs
  
  TI46 isolated-word corpus
    The 46-word vocabulary consists of two sub-vocabularies: 
    (1) the TI 20-word vocabulary (consisting of the digits zero through nine 
        plus the words "enter," "erase," "go," "help," "no," "rubout," "repeat," "stop," "start," and "yes" as well as 
    (2) the TI 26-word "alphabet set" (consisting of the letters "a" through "z").
  
  english isolated word speech dataset
    RM1
    TIMIT Acoustic-Phonetic Continuous Speech Corpus (NIST Speech Disc 1-1)
    Nationwide Speech Project (NSP) corpus : http://www.ling.ohio-state.edu/~clopper.1/nsp/
      CVC Words (N=76) 	mice, dome, bait  (== Consonant-Vowel-Consonant words)
      Targeted Interview Speech (N=10 target words) sleep, shoes, math
    NYNEX PhoneBook: a phonetically-rich isolated-word telephone-speech database
    Medium Vocabulary Urdu Isolated Words Balanced Corpus for Automatic Speech Recognition
    ICSI Meeting Recorder Digits Corpus
    CCW17 Corpus (WUW Corpus)	
  
  LibriSpeech
    http://www.openslr.org/12/
  
  http://www.openslr.org/resources.php
    
  
  https://oscaar.ci.northwestern.edu/overview.php
    Huge list of (requestable) downloads
  
  Corpus Information.ppt - My FIT (my.fit.edu)
    https://www.google.com.sg/url?sa=t&rct=j&q=&esrc=s&source=web&cd=32&ved=0ahUKEwi867ie0NHSAhVFp48KHaqUDAo4HhAWCBswAQ&url=http%3A%2F%2Fmy.fit.edu%2F~vkepuska%2Fece5526%2FCorpus%2520Information.ppt&usg=AFQjCNGDQQNq-QWNXrU0K0UvXPlz6LfYow&sig2=MpKylZ7SVSFPzIEHJ2aeSg&bvm=bv.149397726,d.c2I

  Several useful datasets cited here :
    OLD : https://github.com/pannous/caffe-speech-recognition  (has dataset links)
    NEW : https://github.com/pannous/tensorflow-speech-recognition/
      https://github.com/pannous/tensorflow-speech-recognition/issues/33  :: Need to generate spoken_words_wav.tar on Mac
      http://pannous.net/files/
        http://pannous.net/files/?C=S;O=A

  https://sourceforge.net/projects/currennt/
  
  https://www.quora.com/As-of-2016-which-is-the-best-text-to-speech-application-available-for-Linux
  
  is there a text that covers the entire english phonetic range
    https://www.quora.com/Is-there-a-text-that-covers-the-entire-English-phonetic-range/answer/Sheetal-Srivastava-1
    http://linguistics.stackexchange.com/questions/9315/does-sample-text-exist-that-includes-most-sounds-represented-by-the-internationa
      https://en.wikipedia.org/wiki/The_North_Wind_and_the_Sun
      http://videoweb.nie.edu.sg/phonetic/courses/aae103-web/wolf.html
    Phoneme set:
      http://www.auburn.edu/~murraba/spellings.html
  
    
  TTS : 
    Festival (plus other voices) : = Pretty old tech, piled high...
      https://www.quora.com/Is-there-any-more-soothing-speech-synthesis-program-for-Linux
    Merlin : = Newer approach from Edinburgh
      http://jrmeyer.github.io/merlin/2017/02/14/Installing-Merlin.html
      http://www.cstr.ed.ac.uk/downloads/publications/2010/king_hmm_tutorial.pdf
    Microsoft API :
      https://www.microsoft.com/cognitive-services/en-us/speech-api
        Have 0-9 in all en-* voices
    NeoVoice.com
    Google TTS?

  MFCC
    http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
    
    python_speech_features : Dedicated functionality
      https://github.com/jameslyons/python_speech_features
      pip install python_speech_features
    
    https://github.com/cmusphinx/sphinxtrain/blob/master/python/cmusphinx/mfcc.py

    talkbox : UNMAINTAINED 
      http://scikits.appspot.com/talkbox
      https://github.com/cournape/talkbox

    librosa (has a 'beat-detection' talents too):
      https://github.com/librosa/librosa
      http://librosa.github.io/librosa/feature.html    
  
    pyAudioAnalysis - collection of files...
      https://github.com/tyiannak/pyAudioAnalysis
  
    "Bob" - kitchen sink signal processing research project
      https://www.idiap.ch/software/bob/docs/releases/last/sphinx/html/index.html#
  
    Plus generative (GREAT notebook example): 
      https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    
    And Dynamic Time warping:
      http://nbviewer.jupyter.org/github/crawles/dtw/blob/master/Speech_Recognition_DTW.ipynb
      
      
    https://keunwoochoi.wordpress.com/2016/11/18/for-beginners-writing-a-custom-keras-layer/      

  TODO : 
    DONE  Have function to pull whole of dataset predictions into plain array
    DONE    Collect output of iterator into plain array
    DONE  Graphical/heatmap display of probs
    DONE  animals_ dataset and test set
    DONE  copy into VM : presentations folder 
    
    DONE  Graphical/heatmap display of logits
    
    DONE  animals-test heatmap etc
    DONE  animals training data SVM trick
    
    OKAY search through VM, looking for files/directories to kill with 'clean-up' script
    
    ConvNetJS: MNIST problem fix
      // IDEA : http://stackoverflow.com/questions/6150289/how-to-convert-image-into-base64-string-using-javascript
      // Or : http://stackoverflow.com/questions/16918602/how-to-base64-encode-image-in-linux-bash-shell

    Hat-tip to :
      https://www.html5rocks.com/en/tutorials/canvas/imagefilters/
      
     rsync -avz --progress --exclude 'vm-images*' --exclude 'env3*' --exclude 'VCTK-Corpus*' andrewsm@simlim:~/OpenSource/deep-learning-workshop .
 
      
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


### Microsoft Conference "APAC Machine Learning & Data Science Community Summit" (Seoul) - (2017-05-20)  45mins

This is a 500 Pax multi-track public event co-organized by 5 leading Machine Learning and Data Science related communities in Korea. 

The audience is ordinary community members or developers who are interested in the area of ML and DS. 
They may be beginners to practitioners who uses DS and ML in their day job.

Would appreciate a 45 mins session "regular talk" on Deep Learning and we would love for you to show code. 
There will be 5 mins at the end of your session for Q&A.  
The audience is diverse so we don"t necessarily have to zero in on Tensorflow. 
I think using AlphaGo to bring out reinforcement learning and how it can be applied to things 
beyond games would be a hit for the audience. 
Live demo and codes are critical elements to excite the audience for sure.

 

I hope the input is useful. If you need additional information, please do not hesitate to let us know.

Please do feel free to propose your session title and synopsis. We look forward to having you with us in Seoul!

Learning Korean :
  From TED: 
    # https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus
    # https://github.com/ajinkyakulkarni14/How-I-Extracted-TED-talks-for-parallel-Corpus-
      * https://www.ted.com/talks?sort=newest&q=ai
        * https://www.ted.com/talks/daniel_wolpert_the_real_reason_for_brains/transcript?language=en
        * https://www.ted.com/talks/daniel_wolpert_the_real_reason_for_brains/transcript?language=ko
        * https://www.ted.com/talks/daniel_wolpert_the_real_reason_for_brains/transcript?language=ja

  Learning Japanese (extra) :
    https://pypi.python.org/pypi/tinysegmenter
      https://git.tuxfamily.org/tinysegmente/tinysegmenter.git/tree/src/tinysegmenter.py
    
  From 'Frozen':
    # Dual Subtitles as Parallel Corpora
    # http://www.cs.cmu.edu/~lingwang/papers/lrec2014.pdf

    # Idea for significantly reduced corpus
    # https://cs224d.stanford.edu/reports/GreensteinEric.pdf


CIFAR10:
  square:
    Train Epoch: 1 [   200/ 50000 (0%)]	Loss: 2.3028	t_epoch: 4748.06secs
    Train Epoch: 1 [  4200/ 50000 (8%)]	Loss: 2.1638	t_epoch: 4526.70secs

  Meh - I can't get it to overlearn...


Text output:
  Want to look at the words-from-indices idea 
    Neural Machine Translation via Binary Code Prediction
      https://arxiv.org/abs/1704.06918
      
  Can test using input of features-of-picture-to-be-captioned
    Overview
      http://bengio.abracadoudou.com/publications/pdf/vinyals_2016_pami.pdf
        Has standard capitioning image (should re-screenshot this)
      http://sidgan.me/technical/2016/01/09/Exploring-Datasets
      
    Flickr30k
      Filled in form at :  https://illinois.edu/fb/sec/229675
      Downloaded .tar : 4.4Gb
        Problem  : Can't be stored on vfat thumb-drive
        Solution : tar -xf *.tar ; rsync -avz 
                   But this takes quite a long time...
      Next steps : 
        Run this through inception_v3 (?) to create features
    Flickr8k 
    MS-COCO
      Seems to feature 'mixed action' images (no obvious subject)
      http://mscoco.org/dataset/#download
      https://github.com/pdollar/coco
      
  Pre-built one-hot models (for adaptation) :
    https://github.com/tensorflow/models/tree/master/im2txt/im2txt
      This is the reference implementation for vinyals_2016_pami.pdf
      Pure im2seq (by the paper authors)
    https://github.com/jazzsaxmafia/show_and_tell.tensorflow/blob/master/Readme.md
      flickr30k 
      Pure TF
      No attention
      Expects VGG-16 as featurisations
    https://github.com/LemonATsu/Keras-Image-Caption
      MSCOCO
      SOTA results (for a 2016/17 competition)
      Has attention too
      Expects Inception-V3 as featurisations @ tensor_name='pool_3:0'
      Used GloVe 6bn 100d embeddings 
      
  Rather than LSTM processing, try for aCNN over the text (like Facebook)
    Except that attention doesn't seem to be necessary
    https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f
    
    CNN for decoding : 
      https://github.com/paarthneekhara/byteNet-tensorflow
      https://github.com/Kyubyong/bytenet_translation
    
  
  Want to see which has faster training (and smaller networks)
    In each case feed back embedding as next input :
      Output standard 1-hot encoding 
      Output 511 top words and rest as a word index (possibly with error-correction)
      Output 511 top words and rest as a word embedding, and look for closest match
  
  LSTM-versions:  ( https://github.com/jonsafari/nmt-list )
    https://github.com/lvapeab/nmt-keras
    https://github.com/farizrahman4u/seq2seq  # Understandable code


  CNN sentiment classification (Keras)
    https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras


### TensorFlow and Deep Learning MeetUp talk - (2017-05-25)  30mins

Focus is on Text.  Essentially, 'my turn' to do beginner topic.

Perhaps :

*  Rework the NER thing to TensorFlow (or Keras); or
*  Make the adversarial dictionary word generator/detector thing work


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
  

### Future idea : Tacotron as inspiration...

https://www.reddit.com/r/MachineLearning/comments/62c1on/r_tacotron_a_fully_endtoend_texttospeech/
  Kyle Kastner is in the Montreal lab, and worked on char2wav there : https://github.com/sotelo/parrot
  
Need to go from mel spectrogram -> pure spectrogram
  EXCEPT : SOME PARTS ARE MSR-LA License :: DO NOT READ
    https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe#file-audio_tools-py-L3923  # License: BSD 3-clause

    

  spectrum -> WAV
    Griffin & Lim 1984 :
      https://pdfs.semanticscholar.org/14bc/876fae55faf5669beb01667a4f3bd324a4f1.pdf
    
    Thesis including psuedo-code
      http://iem.kug.ac.at/fileadmin/media/iem/projects/2012/hollomey.pdf
    
    http://ant-s4.unibw-hamburg.de/dafx/paper-archive/2016/dafxpapers/03-DAFx-16_paper_02-PN.pdf
      - links to LTFat (GPL3 )
        http://ltfat.github.io/

    http://perraudin.info/publications/perraudin-note-002.pdf    
      - FGLA
      - links to LTFat
    http://dsp.stackexchange.com/questions/2757/inverse-short-time-fourier-transform-algorithm-described-in-words
      - has some code (MatLab)
      
    http://lonce.org/home/Publications/publications/2007_RealtimeSignalReconstruction.pdf
    

  Looks VERY relevant : https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/
    https://github.com/DmitryUlyanov/neural-style-audio-tf      
    
  python_speech_features  (only featurisation) :
    https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py#L35
    
  librosa has STFT and iSTFT
    https://github.com/librosa/librosa/blob/master/librosa/core/spectrum.py#L181
  
  https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
  https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/sigproc.py
  
  https://github.com/librosa/librosa/blob/master/librosa/core/spectrum.py
  
  https://www.reddit.com/r/MachineLearning/comments/5668yr/for_generative_modelling_on_audio_spectrograms/
  https://github.com/vadim-v-lebedev/audio_style_tranfer/blob/master/audio_style_transfer.ipynb
  
  https://github.com/Kyubyong/tacotron
  

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


## For TF&DL (2017-07-20) "Tips and Tricks":
  Whole span of stuff (in Keras?)
    Beginner, day-to-day, research
  Define model once, with a parameter or two to vary
  Run v. small examples with co-prime dimensionality
  Building model, include a 'probe' in the output vector
    remove once model is done
  Is preprocessing 'finite'
    Do most preprocessing and save to SSD ahead of time (even if multi-GB)
      Rather than do it on-the-fly in Python 
  Keras : the point of fit_generator 
  Check on GPU occupancy : Low : batchsize
  NumPy constant into Keras trick
  Converting Slim to Keras (notebook)
  
  
  
