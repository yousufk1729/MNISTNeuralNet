# MNISTNeuralNet
Simple ANN implementation from scratch in C to recognize handwritten digits from the MNIST dataset. 

## Motivation
I still have no real idea how this works or why it works, but in this experiment I at least tried to use some specific application techniques so I feel smarter than if I just wrote 10 lines with PyTorch. For this reason, I am not trying to write this document as an educational piece but rather as notes I am taking for myself. 

I think enough people gloss over the details regarding neural network math while maintaining to themselves that this is a “high-level” overview:
- Neural networks have neurons with activations, weights, and biases that feed into other neurons, sorted into layers
- Activations introduce non-linearity so the networks can approximate functions like XOR
- Feedforward is when you pass data from input to output 
- Gradient descent is when you try to minimize the error of your network using the gradient (!)
- Backpropagation uses the chain rule to find the gradient
- Stochastic gradient descent involves picking out mini-batches from your data for some computational improvements (drunk man walking down a hill vs careful planning of the optimal route)

Each one of these concepts has such incredible complexity behind it that I think it's a disservice to describe these ideas so succinctly. But, other sources are able to describe the math better, and I have the ones I used in the references. I simply copied their approaches:

## Overview
This model is just for experimentation: No excessive attention is paid to hyperparameters, weight/bias initialization, cost functions, activation functions, hidden layer design, splitting training set to have a validation set, regularization, etc.  I think that effort should be saved for less pedestrian beginner deep learning projects. Model typically achieves ~93% accuracy. 

The program will parse the MNIST dataset. There is enough info online about how the MNIST set is stored, but the most notable thing is that MNIST data is big-endian while my x64 machine (as well as literally every other processor I know of) is little-endian, so my program flips the bytes. Also, this was my first time working with the FILE datatype. 

Weights are initialized with He uniform and biases are set to 0. The program also has the option to load pretrained model parameters stored in a binary file.

Input layer a[0] has 784 neurons (28x28 input) normalized between 0 and 1. 

Hidden layer a[1] has 16 neurons with ReLU activation.

Output layer a[2] has 10 neurons with softmax activation.

MSE is used for cost and ANN “learns” with stochastic gradient descent (includes backpropagation), batch size of 20, learning rate 0.5, and 25 epochs. It trains pretty fast too (around 10-15 seconds with compiler optimization flags). 

The final model parameters are stored in a binary file. 

## References
- https://www.youtube.com/watch?v=tIeHLnjs5U8 
- http://neuralnetworksanddeeplearning.com/chap2.html 
- https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras 
- https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform  
- https://www.kaggle.com/datasets/hojjatk/mnist-dataset 
- https://medium.com/theconsole/do-you-really-know-how-mnist-is-stored-600d69455937   

