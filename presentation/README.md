## Welcome to "deep-learning-workshop"

Please browse to "index.html" in this folder - it's a lot easier to navigate from there.


------------------------------

## Building the Presentation

```
cd ../presentation/
wget https://github.com/hakimel/reveal.js/archive/2.6.2.tar.gz
tar -xzf 2.6.2.tar.gz --exclude=index.html
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
