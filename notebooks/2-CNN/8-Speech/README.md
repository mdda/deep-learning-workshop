## CNNs for Speech Recognition

The steps for running this example are : 

*  Record some sound files that you want to categories (the idea here being an MNIST-for-speech)
*  Normalise and package up the data into a 'proper dataset'
*  Train a speech recognition CNN on the dataset, and test how well the model works
*  Train an SVM layer added on the digits model to learn to classify animal (or whatever) words as an example of transfer learning

The first two steps above are addressed in ```SpeechRecognition_Data.ipynb```, while the training part is in ```SpeechRecognition_Learn.ipynb```.

So far, the ```SpeechSynthesis_Test.ipynb``` isn't in a useable state - 
but the idea is to make something related to Tacotron, starting with simplifying the 
output stage.  The current notebook explores whether sounds can be decently reconstructed
from spectrograms.  Since that seems to work (but by a nasty iterative FFT method), the
next step is to learn to do the transformation via a network, potentially trained
adversarially.


### Layout of Data

```REPO/notebooks/2-CNN/8-Speech/```
*  ```notebooks.ipynb```
*  ```data/```
   *   ```num/``` (digits get sorted into separate subdirectories here)
       *   ```one/```
       *   ```two/```
       *   ```three/```
       *   ...
   *  ```num_ABC.wav```  (these files have digits 0,1,2,3,...9 in order in it)
   *  ```num_XYZ.ogg```
   
   *  ```num-test.pkl```
   *  ```num.pkl```
       
   *   ```animals/```  (animals get sorted into separate subdirectories here)
       *   ```cat```
       *   ```dog```
       *   ```fox```
       *   ...
   *  ```animals_ABC.wav```  (these files have 'cat dog fox bird' in them)
   *  ```animals_XYZ.ogg```

   *  ```animals-test.pkl```
   *  ```animals.pkl```
       
Using the illustration above, the ```data``` directory starts with sound files 
with a series of silence-separated words in them.  These are then normalised, chopped up,
and then shuffled into directories that contain short ```.wav``` files of single word types.

Once that's done, the individual words are then 'imaged' (i.e. converted into spectra) and 
pickled together into the datasets ```num.pkl``` and ```num-test.pkl``` (and similarly for ```animals```).


### Prebuilt datasets

These can be downloaded from the Red Cat Labs server.
<!--
  971  cd notebooks/2-CNN/8-Speech/
  973  tar -czf data-num-sounds.tar.gz data/num_* data/*raw-from-phone.wav
  974  tar -tzf data-num-sounds.tar.gz 
  975  tar -czf data-animals-sounds.tar.gz data/animals_*
  
  977  tar -czf data-num-pkl.tar.gz data/num*.pkl
  978  tar -tzf data-num-pkl.tar.gz 
  979  tar -czf data-animals-pkl.tar.gz data/animals*.pkl
!-->

Sound files (simply recorded on my Android phone using ```Voice Recorder``` in the Android store):

*  <a href="http://redcatlabs.com/downloads/deep-learning-workshop/notebooks/2-CNN/8-Speech/data-num-sounds.tar.gz">```data-num-sounds.tar.gz```</a>
*  <a href="http://redcatlabs.com/downloads/deep-learning-workshop/notebooks/2-CNN/8-Speech/data-animals-sounds.tar.gz">```data-animals-sounds.tar.gz```</a>

and the corresponding processed pickle files (for the really lazy):

*  <a href="http://redcatlabs.com/downloads/deep-learning-workshop/notebooks/2-CNN/8-Speech/data-num-pkl.tar.gz">```data-num-pkl.tar.gz```</a>
*  <a href="http://redcatlabs.com/downloads/deep-learning-workshop/notebooks/2-CNN/8-Speech/data-animals-pkl.tar.gz">```data-animals-pkl.tar.gz```</a>

These should be unpacked (```tar -xzf XYZ.tar.gz```) in the ```REPO/notebooks/2-CNN/8-Speech/``` directory.


### Training Process

The ```SpeechRecognition_Data.ipynb``` notebook has several parts : 

*  Play with the normalization and segmentation of specific files in the ```data/``` directory
*  Normalize and split into individual words all sound files with a specific prefix (eg: ```num_```) in the ```data/``` directory
*  For all the words now in their individual directories (eg: ```data/num/six/```), 
   form the spectrograms (also called 'stamps' here, since they look like stamp-sized pictures), and 
   bundle them together in datasets as ```.pkl``` files

This process happens for both ```num_``` and ```animals_``` data.

If you'd like to record your own sound files (or different words, etc), 
just have a look/listen to some of the examples.  It is a deliberately low-tech setup.

Having generated the ```.pkl``` files, you can switch attention to  ```SpeechRecognition_Learn.ipynb```.

In this notebook, a simple CNN (adapted from something that works for MNIST) is trained up on the numbers.  And
then tested to make sure the digits are recognised accurately.  Note that then number of training examples
is *extremely* low (around 20 examples of each digit, of which ~2 are withheld for the test set).

Having accomplished the main 'aha' learning, there's an extra trick in the next section : Instead of 
just recognising the digits, let's use the network trained on the numbers to recognise other words.

The way this is done is by 'featurising' some additional word examples, by passing the stamps through the
digit recogniser, and then looking at the logit outputs *as if* they were digits.  Then, using those 
outputs as features, training a separate classifier, and seeing whether that classifier can be used to
correctly recognise (in this case) animal words.

Again, the amount of data is really low (perhaps 4-5 examples of each animal).  So the main take-away should be
that this technique works surprisingly well (i.e. we should be surprised that it works at all).





