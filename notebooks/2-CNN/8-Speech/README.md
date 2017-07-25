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

*  notebooks.ipynb
*  data/
   *   num/ (digits get sorted into separate subdirectories here)
       *   one/
       *   two/
       *   three/
       *   ...
   *  num_ABC.wav  (these files have digits 0,1,2,3,...9 in order in it)
   *  num_XYZ.ogg
   
   *  num-test.pkl
   *  num.pkl
       
   *   animals/  (animals get sorted into separate subdirectories here)
       *   cat
       *   dog
       *   fox
       *   ...
   *  animals_ABC.wav  (these files have 'cat dog fox bird' in them)
   *  animals_XYZ.ogg

   *  animals-test.pkl
   *  animals.pkl
       
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



and the corresponding processed pickle files (for the really lazy):






