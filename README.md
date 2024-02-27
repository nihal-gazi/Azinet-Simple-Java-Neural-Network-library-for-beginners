# Azinet: Simple Java Neural Network library for beginners
Simple and easy-to-use neural network library made into a SINGLE class. Just copy paste into a class, and start using it!

The Neural Network.java contains 2 class: NeuralNetwork and Matrix. The matrix class is used for neural network calculations. The neural network class contains all necessary functions.

A NeuralNetworkDriver.java class is there which demonstrates how to use the library.

[Note: For better readability, the codes are not optimized]

## NeuralNetwork.java
### Initialization
A network is initialized by accepting an array of nodes. For example, a neural network with 3 input nodes, 5 nodes in hidden layers, 4 nodes in next hidden layer and 1 node in output layer can be written as:
```java
int[] structure = {2, 3,3,1};
```
To begin, make an object of Neural Network
```java
int[] structure = {2, 3,3,1};
NeuralNetwork neuralNetwork = new NeuralNetwork(structure, true);
```
The boolean arguement next to the `int[] structure` is whether you want to initialize weights of network with random values or not.
if `false`, then all weights will be initialized to 0.



