# Azinet: Simple Java Neural Network library for beginners
Simple and easy-to-use neural network library made into a SINGLE class. Just copy paste into a class, and start using it!

**No need for advanced knowledge. Not even calculus is required. This library is made keeping in mind that it is understandable to high school students.**

The Neural Network.java contains 2 class: NeuralNetwork and Matrix. The matrix class is used for neural network calculations. The neural network class contains all necessary functions.

A NeuralNetworkDriver.java class is there which demonstrates how to use the library.

[Note: For better readability, the codes are not optimized]

## NeuralNetwork.java
### Initialization
A network is initialized by accepting an array of nodes. For example, a neural network with 2 input nodes, 3 nodes in hidden layers, 3 nodes in next hidden layer and 1 node in output layer can be written as:
```java
int[] structure = {2, 3,3,1};
```
To begin, make an object of Neural Network
```java
int[] structure = {2, 3,3,1};
NeuralNetwork neuralNetwork = new NeuralNetwork(structure, true);
```
The boolean arguement next to the `int[] structure` is whether you want to initialize weights of network with random values or not.
If `false`, then all weights will be initialized to 0.

------------


### Dataset
Make 2 separate double 2D array: `input` and `target`. 

for example, a XOR dataset looks like this:
```java
double[][] inputs = {
            {0,0},
            {1,0},
            {0,1},
            {1,1}
        };

        double[][] targets = {
            {0},
            {1},
            {1},
            {0}
        };
```


------------


### Training
For keeping things simple, we will use stochastic gradient descent. 
> Stochastic Gradient Descent: Update weights to reduce error, and sometimes add random offsets to weight updation for the network to not get stuck in local optima.

Training algorithm:
1. Change weights by adding and subtracting a value `alpha` to a weight.
2. Calculate and store total error in each case.
3. If adding or subtracting `alpha` reduces the error, then do the respective modification, else keep the weights unchanged.
4. At random probability(as set by user in `update_probability`), we will add small random perturbation to the weight, to bring variation so as to prevent the net from getting stuck at local optima.

> Note: If the error of the network is getting stagnant at each epoch, then the `alpha` is updated by dividing it by 10, to fine tune the learning.

Code:

This function is used for training:
```java
train_stochastic(double[][] input, double[][] output, int epochs, double alpha, double update_probability)
```
To train the AI on our XOR dataset, we will write:
```java
neuralNetwork.train_stochastic(inputs, targets, 300,100, 1);
```
`300` is the value of maximum epochs(iterations of training).
`100` is the learning rate, i.e. `alpha`.
`1` is the `update_probability`.

### Run the Neural Network
To run the neural network, first it needs some input. The inputs are accepted in 1D double array. Lets say, we want to run our network on the inputs '0' and '1'.
So, we will write:
```java
double[] testInput = {0,1};
 double[] prediction = neuralNetwork.feedForward(testInput);
```
The `feedForward` function is used to run the net. The net outputs a double array of output values. The number of output values depend on number of output nodes. 1 output node should give an output array containing 1 element.





