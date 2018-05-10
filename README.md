# tensor-fashion
Exploring the fashion MNIST using tensorflow

Here's a link to the complete dataset on kaggle.com
https://www.kaggle.com/zalando-research/fashionmnist/data


Fashion mnist is a dataset of 28x28 pixel grayscale images of clothes belonging to 10 catagories.    
classes=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle-boot']

The total number of images is 70,000 with the train and test sets being divided in a 6:1 ratio.

The architecture used is a neural network having 3 hidden layers. 
The input layer has 28x28=784 neurons. 
The output layer has 10 neurons one for each class.

The neurons in the hidden layers are 200, 100 and 30.
Xavier initialization was used to initialize the weights.
The optimizer used was the Adam optimizer of tf.train .

a train accuracy of 0.978 was achieved.
test accuracy achieved is 0.888.

a tutorial to the whole process can be found in this blog.
[link]


