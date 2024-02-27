import java.util.*;
public class NeuralNetworkDriver {
    public static void main() {
        // Define the structure of the neural network (3 input nodes, 2 hidden layers, 1 output node)
        int[] structure = {2, 3,3,1};

        // Create a neural network with the specified structure
        NeuralNetwork neuralNetwork = new NeuralNetwork(structure, true);
        

        // Sample training data (2D arrays where each row represents a training sample)
        //current dataset: XOR
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
