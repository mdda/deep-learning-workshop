### Building the Presentation

```
cd ../presentation/
wget https://github.com/hakimel/reveal.js/archive/2.6.2.tar.gz
tar -xzf 2.6.2.tar.gz --exclude=index.html
```

Open in Chrome : 
```
<PATH-TO-REPO>/presentation/index.html
```


#### Adding ConvNetJS examples 

```
#http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html

cd ../presentation/
wget http://cs.stanford.edu/people/karpathy/convnetjs/build/convnet-min.js


# Both need:
<link rel="stylesheet" href="css/style.css">

<script src="js/jquery-1.8.3.min.js"></script>

<script src="../build/convnet.js"></script>


# image_regression needs : 
<link href='http://fonts.googleapis.com/css?family=Cabin' rel='stylesheet' type='text/css'>
<link rel="stylesheet" href="css/jquery-ui.min.css">

<script src="js/jquery-ui.min.js"></script>

<script src="js/image_regression.js"></script>


# MNIST needs :
<link rel="stylesheet" href="css/style.css">

<script src="../build/vis.js"></script>
<script src="../build/util.js"></script>

<script src="js/image-helpers.js"></script>
<script src="js/pica.js"></script>

<script src="mnist/mnist_labels.js"></script>



```

