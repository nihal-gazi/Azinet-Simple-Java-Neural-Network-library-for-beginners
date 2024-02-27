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

for example, a XOR gate dataset looks like this:
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
`1` is the `update_probability`. (We are setting it to 1 to avoid stochastic perturbation for simplicity)

------------


### Run the Neural Network
To run the neural network, first it needs some input. The inputs are accepted in 1D double array. Lets say, we want to run our network on the inputs '0' and '1'.
So, we will write:
```java
double[] testInput = {0,1};
double[] prediction = neuralNetwork.feedForward(testInput);
```
The `feedForward` function is used to run the net. The net outputs a double array of output values. The number of output values depend on number of output nodes. 1 output node should give an output array containing 1 element.

------------

### Output
Our final code looks like this:
```java
import java.util.*;
public class NeuralNetworkDriver {
    public static void main() {
        // Define the structure of the neural network (3 input nodes, 2 hidden layers, 1 output node)
        int[] structure = {2, 3,3,1};

        // Create a neural network with the specified structure
        NeuralNetwork neuralNetwork = new NeuralNetwork(structure, true);
        

        // Sample training data (2D arrays where each row represents a training sample)
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
        

        neuralNetwork.train_stochastic(inputs, targets, 300, 1000.0 , 1); 
        
        //show weights and biases
        neuralNetwork.print_weights();
        neuralNetwork.print_biases();

        // Test the trained network with some input data
        double[] testInput = {0,1};
        double[] prediction = neuralNetwork.feedForward(testInput);

        System.out.println("Prediction for input "+Arrays.toString(testInput)+": " + prediction[0]);
        
        double[] testInput2= {1,1};
        double[] prediction2 = neuralNetwork.feedForward(testInput2);

        System.out.println("Prediction for input "+Arrays.toString(testInput2)+": " + prediction2[0]);
    }
}

```
Running this, should give you an output similar to this:
```
NeuralNetworkDriver.main();
Best error=0.004954059763381859		epoch=1
Best error=5.266613918547254E-9		epoch=2
Best error=5.266613918547254E-9		epoch=3
Best error=1.9592203964956884E-52		epoch=4
Best error=7.288448728733523E-96		epoch=5
Best error=2.7113583018221186E-139		epoch=6
Best error=1.0086458880993115E-182		epoch=7
Best error=3.7522393366304484E-226		epoch=8
Best error=1.3958615412479291E-269		epoch=9
Best error=1.3958615412479291E-269		epoch=10
Best error=1.3958615412479291E-269		epoch=11
Best error=5.135087637469963E-270		epoch=12
Best error=1.889093170438832E-270		epoch=13
Best error=6.949585398618258E-271		epoch=14
Best error=2.5566095928168995E-271		epoch=15
Best error=9.405241082990298E-272		epoch=16
Best error=3.459994833693162E-272		epoch=17
Best error=1.272860965875118E-272		epoch=18
Best error=4.682593808150807E-273		epoch=19
Best error=4.682593808150807E-273		epoch=20
Best error=4.236986091078251E-273		epoch=21
Best error=3.833783554905431E-273		epoch=22
Best error=3.4689508131292734E-273		epoch=23
Best error=3.138836497045563E-273		epoch=24
Best error=2.8401367116236784E-273		epoch=25
Best error=2.8401367116236784E-273		epoch=26
Best error=2.811876879167957E-273		epoch=27
Best error=2.783898236743391E-273		epoch=28
Best error=2.7561979864624214E-273		epoch=29
Best error=2.728773358276937E-273		epoch=30
Best error=2.701621609701265E-273		epoch=31
Best error=2.6747400255379213E-273		epoch=32
Best error=2.6747400255379213E-273		epoch=33
Best error=2.6720666224367805E-273		epoch=34
Best error=2.6693958914024855E-273		epoch=35
Best error=2.6667278297643033E-273		epoch=36
Best error=2.6640624348541733E-273		epoch=37
Best error=2.6613997040067E-273		epoch=38
Best error=2.6613997040067E-273		epoch=39
Best error=2.661133577342921E-273		epoch=40
Best error=2.660867477290478E-273		epoch=41
Best error=2.6606014038467098E-273		epoch=42
Best error=2.6606014038467098E-273		epoch=43
Best error=2.660574797965768E-273		epoch=44
Best error=2.6605481923508842E-273		epoch=45
Best error=2.6605215870020546E-273		epoch=46
Best error=2.6604949819192773E-273		epoch=47
Best error=2.66046837710255E-273		epoch=48
Best error=2.6604417725518687E-273		epoch=49
Best error=2.660415168267232E-273		epoch=50
Best error=2.660415168267232E-273		epoch=51
Best error=2.6604125078534007E-273		epoch=52
Best error=2.6604098474422298E-273		epoch=53
Best error=2.6604071870337194E-273		epoch=54
Best error=2.660404526627869E-273		epoch=55
Best error=2.660401866224679E-273		epoch=56
Best error=2.66039920582415E-273		epoch=57
Best error=2.6603965454262816E-273		epoch=58
Best error=2.6603965454262816E-273		epoch=59
Best error=2.6603962793867314E-273		epoch=60
Best error=2.660396013347208E-273		epoch=61
Best error=2.6603957473077115E-273		epoch=62
Best error=2.6603954812682417E-273		epoch=63
Best error=2.6603952152287983E-273		epoch=64
Best error=2.6603949491893815E-273		epoch=65
Best error=2.6603949491893815E-273		epoch=66
Best error=2.660394922585411E-273		epoch=67
Best error=2.660394922585411E-273		epoch=68
Best error=2.660394922585411E-273		epoch=69
Best error=2.6603949223192534E-273		epoch=70
Best error=2.660394922053096E-273		epoch=71
Best error=2.6603949217869376E-273		epoch=72
Best error=2.66039492152078E-273		epoch=73
Best error=2.6603949212546223E-273		epoch=74
Best error=2.660394920988465E-273		epoch=75
Best error=2.6603949207223074E-273		epoch=76
Best error=2.6603949204561495E-273		epoch=77
Best error=2.6603949204561495E-273		epoch=78
NeuralNetwork.java : [Maximum accuracy achieved. Training stopped.]

----------------------------------------
	WEIGHTS:

Weight matrix from layer 0 to layer 1
-12.356207420851007	-1016.512050254621	
-7.850312847925622	1013.0466734578782	
-1008.8175124287047	12.907993255628135	

Weight matrix from layer 1 to layer 2
-992.6774577487947	-988.8111996744096	1002.6735480771881	
-1016.7424937218094	-1003.4093562211723	15.477430991786981	
971.2066623104607	1013.0404557379202	-1000.5278649888202	

Weight matrix from layer 2 to layer 3
996.9420729406887	25.057679930632887	-628.320402989824	

----------------------------------------


----------------------------------------
	BIASES:

Bias matrix from layer 0 to layer 1
0.0	
0.0	
0.0	

Bias matrix from layer 1 to layer 2
0.0	
0.0	
0.0	

Bias matrix from layer 2 to layer 3
0.0	

----------------------------------------

Prediction for input [0.0, 1.0]: 1.0
Prediction for input [1.0, 1.0]: 1.3301974602280748E-273
```
We find that the outputs are very close to our dataset. `2.6603949204561495E-273` means 2.66×10^-273 which is a very small error indicating that out training was done very well!


**But sometimes, the network may not train successfully, such as this example.**
Failure case:
```
NeuralNetworkDriver.main();
Best error=1.5		epoch=1
Best error=1.5		epoch=2
Best error=1.5		epoch=3
Best error=1.5		epoch=4
Best error=1.5		epoch=5
Best error=1.5		epoch=6
Best error=1.5		epoch=7
Best error=1.5		epoch=8
Best error=1.5		epoch=9
Best error=1.5		epoch=10
Best error=1.5		epoch=11
Best error=1.5		epoch=12
Best error=1.5		epoch=13
Best error=1.5		epoch=14
Best error=1.5		epoch=15
NeuralNetwork.java : [Maximum accuracy achieved. Training stopped.]

----------------------------------------
	WEIGHTS:

Weight matrix from layer 0 to layer 1
-25.654985069100753	994.8528615997126	
-5.59504798232917	-1.4551338517727217	
21.915198828488435	7.043195628326359	

Weight matrix from layer 1 to layer 2
-989.5821453296771	-10.344679763838485	0.033688608259041075	
-1008.6223719017355	23.795389038249823	-30.64093074235585	
-1027.3316422252321	1.7175238273506963	26.517645836677957	

Weight matrix from layer 2 to layer 3
980.7717364617635	2.600112474450789	21.035167094034684	

----------------------------------------


----------------------------------------
	BIASES:

Bias matrix from layer 0 to layer 1
0.0	
0.0	
0.0	

Bias matrix from layer 1 to layer 2
0.0	
0.0	
0.0	

Bias matrix from layer 2 to layer 3
0.0	

----------------------------------------

Prediction for input [0.0, 1.0]: 0.5
Prediction for input [1.0, 1.0]: 0.5

```

**Failure cases are NORMAL. You may need to run the net more than 1 time to get success.**

------------

## Final Words
> This repository is made with the intention that it is to be used only for educational purposes.
> Constructive Crticism from students and teachers is welcomed, and to be sent at nihalg2006@gmail.com.
> Before sending email, please put the subject to `[Azinet]` in your email, so that I know that it is from you.

------------





