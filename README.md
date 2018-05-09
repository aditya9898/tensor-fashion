# tensor-fashion
exploring the fashion MNIST using tensorflow

fashion mnist is a dataset of 28x28 pixel grayscale images of clothes belonging to 10 catagories.
classes=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle-boot']
the total number of images is 70,000 with the train and test sets being divided in a 6:1 ratio.

the architecture used is a neural network having 3 hidden layers.
the input layer has 28x28=784 neurons.
the output layer has 10 neurons one for each class.

the neurons in the hidden layers are 200, 100 and 30.
Xavier initialization was used to initialize the weights.
the optimizer used was the Adam optimizer of tf.train .

a train accuracy of 0.978 was achieved.
test accuracy achieved is 0.888.

a tutorial to the whole process can be found in this blog.
[link]


