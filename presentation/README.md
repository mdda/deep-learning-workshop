## Welcome to "deep-learning-workshop"

Please browse to "index.html" in this folder - it's a lot easier to navigate from there.


------------------------------

## Building the Presentation

```
cd ../presentation/
wget https://github.com/hakimel/reveal.js/archive/2.6.2.tar.gz
tar -xzf 2.6.2.tar.gz --exclude=index.html
```


```
cd ../presentation/

REVEAL=reveal.js-2.6.2

wget https://fonts.gstatic.com/s/quicksand/v5/sKd0EMYPAh5PYCRKSryvW6CWcynf_cDxXwCLxiixG1c.ttf

wget https://fonts.gstatic.com/s/opensans/v13/cJZKeOuBrn4kERxqtaUH3aCWcynf_cDxXwCLxiixG1c.ttf
wget https://fonts.gstatic.com/s/opensans/v13/k3k702ZOKiLJc3WVjuplzInF5uFdDttMLvmWuJdhhgs.ttf
wget https://fonts.gstatic.com/s/opensans/v13/xjAJXh38I15wypJXxuGMBp0EAVxt0G0biEntp43Qt6E.ttf
wget https://fonts.gstatic.com/s/opensans/v13/PRmiXeptR36kaC0GEAetxp_TkvowlIOtbR7ePgFOpF4.ttf

mkdir -r ${REVEAL}/fonts
mv *.ttf ${REVEAL}/fonts/
```




Open in Chrome : 
```
<PATH-TO-REPO>/presentation/index.html
```


### Adding ConvNetJS example(s) 

Scheme, seeing the number of files included, probably simplest to :

*  Create a copy the repo inside ```./presentation/reveal.js-2.6.2/```
*  Add ```convnet.js.min``` in ```<CONVNET-REPO>/build```
*  Add jQuery from source
*  Add the required font
*  Create copies (for customisation) of relevant pages and add to deep-learning repo
   *  Potentially need to copy in from another location (repos-within-repos problem)

Examples to use : 
*  http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html
*  http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html
*  (instead of MNIST?) : http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html

```
cd ../presentation/

REVEAL=reveal.js-2.6.2
CONVNET=${REVEAL}/convnetjs

wget https://github.com/karpathy/convnetjs/archive/master.zip
unzip master.zip
cp -R convnetjs-master ${REVEAL}/convnetjs
rm -rf convnetjs-master
rm master.zip

DEMO=${CONVNET}/demo
KARPATHY=http://cs.stanford.edu/people/karpathy
wget --directory-prefix=${DEMO}/../build/ ${KARPATHY}/convnetjs/build/convnet-min.js 

//GONE wget --directory-prefix=${DEMO}/js/ ${KARPATHY}/convnetjs/demo/jquery-ui.min.js  # NB: This is relocated
//GONE wget --directory-prefix=${DEMO}/css/ ${KARPATHY}/convnetjs/demo/jquery-ui.min.css  # NB: This is relocated

mkdir -p ${DEMO}/imgs
wget --directory-prefix=${DEMO}/imgs/ ${KARPATHY}/convnetjs/demo/imgs/cat.jpg
wget --directory-prefix=${DEMO}/imgs/ ${KARPATHY}/convnetjs/demo/imgs/earth.jpg
wget --directory-prefix=${DEMO}/imgs/ ${KARPATHY}/convnetjs/demo/imgs/pencils.png
wget --directory-prefix=${DEMO}/imgs/ ${KARPATHY}/convnetjs/demo/imgs/starry.jpg


## But the MNIST example runs into a (larger) problem than even 'cat'

mkdir -p reveal.js-2.6.2/convnetjs/demo/mnist
wget --directory-prefix=${DEMO}/mnist/ ${KARPATHY}/convnetjs/demo/mnist/mnist_labels.js
wget --directory-prefix=${DEMO}/mnist/ ${KARPATHY}/convnetjs/demo/mnist/mnist_batch_0.png
wget --directory-prefix=${DEMO}/mnist/ ${KARPATHY}/convnetjs/demo/mnist/mnist_batch_20.png
... MNIST has the same cross-origin security problem that we had to fix for 'cat' (see below)...

```

#### To build : ```image_regression_custom_images.js```

```
base64 --wrap=0 ${DEMO}/imgs/cat.jpg > cat.base64.txt
```


### Adding Tensorflow Playground

```
wget https://github.com/tensorflow/playground/archive/master.zip
unzip  master.zip 
cp -R playground-master/dist reveal.js-2.6.2/tensorflow-playground
rm -rf playground-master
rm master.zip
```

But also need to fix up some of the fonts, since we may need to run completely disconnected from WiFi :

This is the font link that needs re-jigging:
```
wget 'https://fonts.googleapis.com/css?family=Roboto:300,400,500|Material+Icons'
```

```
mkdir -p reveal.js-2.6.2/tensorflow-playground/fonts/

wget https://fonts.gstatic.com/s/materialicons/v17/2fcrYFNaTjcS6g4U3t-Y5StnKWgpfO2iSkLzTz-AABg.ttf
wget https://fonts.gstatic.com/s/roboto/v15/Hgo13k-tfSpn0qi1SFdUfaCWcynf_cDxXwCLxiixG1c.ttf
wget https://fonts.gstatic.com/s/roboto/v15/zN7GBFwfMP4uA6AR0HCoLQ.ttf
wget https://fonts.gstatic.com/s/roboto/v15/RxZJdnzeo3R5zSexge8UUaCWcynf_cDxXwCLxiixG1c.ttf

mv *.ttf reveal.js-2.6.2/tensorflow-playground/fonts/
```

And finally, substitute the font 'pull' : 
```
#  Change : https://fonts.googleapis.com/css?family=Roboto:300,400,500|Material+Icons
#  to     : fonts/material-icons.css

perl -pi.orig -e 's{https://fonts.googleapis.com/css\?family=Roboto:300,400,500\|Material\+Icons}{fonts/material-icons.css}' reveal.js-2.6.2/tensorflow-playground/index.html

```


  <link href="https://fonts.googleapis.com/css?family=Roboto:300,400,500|Material+Icons" rel="stylesheet" type="text/css">
  <link href="fonts/material-icons.css" rel="stylesheet" type="text/css">


