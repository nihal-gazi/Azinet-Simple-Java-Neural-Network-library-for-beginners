import java.util.*;

public class NeuralNetwork {
    /**
    Library name(ID): "Azinet"

    Simple easy-to-use and plug-and-play Neural Network library by Nihal Gazi
    Date of creation :- 27th February 2024

    (Under MIT Licence)
     */
    Random random = new Random();
    
    
    
    private int[] structure;
    private Matrix[] weights;
    private Matrix[] biases;

    double lowest_weight = -32;
    double highest_weight = 32;

    double lowest_bias = -8;
    double highest_bias = 8;

    boolean update_bias = false;

    boolean initialize_bias = false;

    String activation = "sigmoid";

    // Constructor to create a neural network with the given structure
    public NeuralNetwork(int[] structure, boolean randomInitialize) {
        this.structure = structure;
        this.weights = new Matrix[structure.length - 1];
        this.biases = new Matrix[structure.length - 1];

        // Initialize weights and biases randomly
        for (int i = 0; i < structure.length - 1; i++) {
            weights[i] = new Matrix(structure[i + 1], structure[i]);
            biases[i] = new Matrix(structure[i + 1], 1);
            if(randomInitialize){
                // Initialize weights randomly between -1 and 1
                for (int row = 0; row < weights[i].rows; row++) {
                    for (int col = 0; col < weights[i].columns; col++) {
                        weights[i].data[row][col] = random_double(lowest_weight, highest_weight);
                    }
                }

                // Initialize biases randomly between -1 and 1
                if(initialize_bias){
                    for (int row = 0; row < biases[i].rows; row++) {
                        biases[i].data[row][0] = random_double(lowest_bias, highest_bias);
                    }
                }
            }
        }

    }

    public void randomize_weights_biases(double low, double high){
        for (int i = 0; i < structure.length - 1; i++) {
            weights[i] = new Matrix(structure[i + 1], structure[i]);
            biases[i] = new Matrix(structure[i + 1], 1);

            // Initialize weights randomly between -1 and 1
            for (int row = 0; row < weights[i].rows; row++) {
                for (int col = 0; col < weights[i].columns; col++) {
                    weights[i].data[row][col] = (high-low) * random.nextDouble() + low;
                }
            }

            // Initialize biases randomly between -1 and 1
            for (int row = 0; row < biases[i].rows; row++) {
                biases[i].data[row][0] = (high-low) * random.nextDouble() + low;
            }

        }
    }

    // Tanh activation function
    private double tanh(double x) {
        return Math.tanh(x);
    }

    // Sigmoid activation function
    private double sigmoid(double x) {
        return 1.0/(1 + Math.exp(-x));
    }

    public double calculateTotalError(double[][] inputs, double[][] targets){

        double totalError=0.0;

        for(int i = 0;i<inputs.length;i++){
            double output[] = feedForward(inputs[i]);

            double error=0.0;
            for(int j = 0 ; j < targets[0].length; j++){
                error+= Math.abs( Math.max(targets[i][j],output[j]) -  Math.min(targets[i][j],output[j]));
            }

            totalError += error;
        }
        return -totalError;
    }

    public double calculateRandomError(double[][] inputs, double[][] targets){
        double totalError=0.0;
        int i = random(0,inputs.length-1);
        double output[] = feedForward(inputs[i]);

        double error=0.0;
        for(int j = 0 ; j < targets[0].length; j++){
            error+= Math.abs    ( Math.max(targets[i][j],output[j]) -  Math.min(targets[i][j],output[j]));
        }

        totalError += error;

        return -totalError;
    }

    public void copyNet(NeuralNetwork n1, NeuralNetwork n2) {
        // Copy biases
        for (int i = 0; i < n2.biases.length; i++) {
            for (int x = 0; x < n2.biases[i].rows; x++) {
                for (int y = 0; y < n2.biases[i].columns; y++) {
                    try{
                        n2.biases[i].set(x, y, n1.biases[i].get(x, y));}
                    catch(Exception e){
                        n2.biases[i].set(x, y, 0);
                    }
                }
            }
        }

        // Copy weights
        for (int i = 0; i < n2.weights.length; i++) {
            for (int x = 0; x < n2.weights[i].rows; x++) {
                for (int y = 0; y < n2.weights[i].columns; y++) {
                    try{
                        n2.weights[i].set(x, y, n1.weights[i].get(x, y));}
                    catch(Exception e){
                        n2.weights[i].set(x, y, 0);
                    }
                }
            }
        }
    }

    // Feedforward function to calculate output for given input
    public double[] feedForward(double[] input) {
        if (input.length != structure[0]) {
            System.out.println("Invalid input size.");
            return null;
        }

        // Convert input array to a matrix
        Matrix inputMatrix = new Matrix(structure[0], 1);
        for (int i = 0; i < input.length; i++) {
            inputMatrix.data[i][0] = input[i];
        }

        // Perform feedforward calculations
        Matrix output = inputMatrix;
        for (int i = 0; i < structure.length - 1; i++) {
            output = Matrix.multiply(weights[i], output);
            output = Matrix.add(output, biases[i]);
            // Apply activation function (tanh)
            for (int row = 0; row < output.rows; row++) {
                for (int col = 0; col < output.columns; col++) {
                    if(activation.equals("sigmoid")){
                        output.data[row][col] = sigmoid(output.data[row][col]);
                    }

                    if(activation.equals("tanh")){
                        output.data[row][col] = tanh(output.data[row][col]);
                    }

                    if(activation.equals("relu")){
                        output.data[row][col] = Math.max(output.data[row][col],0);
                    }
                }
            }
        }

        // Convert output matrix to a 1D array
        double[] outputArray = new double[output.rows];
        for (int i = 0; i < output.rows; i++) {
            outputArray[i] = output.data[i][0];
        }

        return outputArray;
    }

    private double calculateLoss(double[] input, double[] actual) {
        double loss = 0;
        double[] predicted = feedForward(input);
        for (int i = 0; i < predicted.length; i++) {
            loss += Math.pow(predicted[i] - actual[i], 2);
        }
        return loss / predicted.length;
    }

    // Tanh derivative for backpropagation
    private double tanhDerivative(double x) {
        return 1-Math.pow(tanh(x),2);
    }

    private int random(int low, int high) {
        return (int)(Math.random()*(high - low) + low +1);
    }

    private double random_double(double low, double high) {
        return (Math.random()*(high - low) + low);
    }

    public void train_stochastic(double[][] input, double[][] output, int epochs, double alpha, double update_probability){
        //alpha is the learning rate

        //update probability is the probability of updating a weight in the network
        //if update_probability = 1.0, it perfectly behaves like gradient descent
        //lower update_probability will allow faster learning but inaccurate results

        double best_error = Double.MAX_VALUE;

        for(int epoch = 1; epoch<=epochs ; epoch++){
            double previous_best_error = best_error; //storing a copy of best error before updation
            for (int i = structure.length - 1 -1; i >=0 ; i--) {

                for (int row = 0; row < weights[i].rows; row++) {
                    for (int col = 0; col < weights[i].columns; col++) {
                        if(Math.random() < update_probability){
                            //store old weight
                            double weight = weights[i].data[row][col];

                            //calculate LHE
                            weights[i].data[row][col] = weight + alpha;
                            double error_L = Math.abs(calculateTotalError(input , output));

                            //calculate RHE
                            weights[i].data[row][col] = weight - alpha;
                            double error_R = Math.abs(calculateTotalError(input , output));

                            //set weights back to normal
                            weights[i].data[row][col] = weight;

                            //compare and set
                            if(error_L<best_error){weights[i].data[row][col] = weight + alpha; best_error = error_L; }
                            if(error_R<best_error){weights[i].data[row][col] = weight - alpha; best_error = error_R;}
                        }
                    }
                }
                if(update_bias){
                    for (int row = 0; row < biases[i].rows; row++) {

                        if(Math.random() < update_probability){
                            //store old bias
                            double bias = biases[i].data[row][0];

                            //calculate LHE
                            biases[i].data[row][0] = bias + alpha;
                            double error_L = Math.abs(calculateTotalError(input , output));

                            //calculate RHE
                            biases[i].data[row][0] = bias - alpha;
                            double error_R = Math.abs(calculateTotalError(input , output));

                            //set biases to normal
                            biases[i].data[row][0] = bias;

                            //compare and set
                            if(error_L<best_error){biases[i].data[row][0] = bias + alpha; best_error = error_L;}
                            if(error_R<best_error){biases[i].data[row][0] = bias - alpha; best_error = error_R;}

                        }
                    }
                }

            }

            System.out.println("Best error="+best_error+"\t\tepoch="+epoch);
            //reduce alpha, if learning is stagnant
            if(Math.random() <= update_probability && (previous_best_error == best_error)){
                if(true){
                    alpha = alpha/10;
                }
                else{
                    alpha = alpha/10;
                }
            }
            if(best_error == 0){
                System.out.println("NeuralNetwork.java : [Best accuracy achieved. Training stopped.]");
                break;
            }
            if(alpha < Math.pow(10,-10)){
                System.out.println("NeuralNetwork.java : [Maximum accuracy achieved. Training stopped.]");
                break;
            }
            // print_weights();
            // print_biases();
        }

    }

    public void train_stochastic_v2(double[][] input, double[][] output, int epochs, double alpha, double update_probability){
        //alpha is the learning rate

        //update probability is the probability of updating a weight in the network
        //if update_probability = 1.0, it perfectly behaves like gradient descent
        //lower update_probability will allow faster learning but inaccurate results

        double best_error = Double.MAX_VALUE;

        for(int epoch = 1; epoch<=epochs ; epoch++){
            double previous_best_error = best_error; //storing a copy of best error before updation
            int sample_index = random(0 , input.length-1);
            for (int i = structure.length - 1 -1; i >=0 ; i--) {

                for (int row = 0; row < weights[i].rows; row++) {
                    for (int col = 0; col < weights[i].columns; col++) {
                        if(Math.random() < update_probability){
                            //store old weight
                            double weight = weights[i].data[row][col];

                            //calculate LHE
                            weights[i].data[row][col] = weight + alpha;
                            double error_L = Math.abs(calculateLoss(input[sample_index] , output[sample_index]));

                            //calculate RHE
                            weights[i].data[row][col] = weight - alpha;
                            double error_R = Math.abs(calculateLoss(input[sample_index] , output[sample_index]));

                            //set weights back to normal
                            weights[i].data[row][col] = weight;

                            //compare and set
                            if(error_L<error_R){weights[i].data[row][col] = weight + alpha;  }
                            else{weights[i].data[row][col] = weight - alpha; }
                        }
                    }
                }
                if(update_bias){
                    for (int row = 0; row < biases[i].rows; row++) {

                        if(Math.random() < update_probability){
                            //store old bias
                            double bias = biases[i].data[row][0];

                            //calculate LHE
                            biases[i].data[row][0] = bias + alpha;
                            double error_L = Math.abs(calculateLoss(input[sample_index] , output[sample_index]));

                            //calculate RHE
                            biases[i].data[row][0] = bias - alpha;
                            double error_R = Math.abs(calculateLoss(input[sample_index] , output[sample_index]));

                            //set biases to normal
                            biases[i].data[row][0] = bias;

                            //compare and set
                            if(error_L<error_R){biases[i].data[row][0] = bias + alpha; }
                            else{biases[i].data[row][0] = bias - alpha; }
                        }
                    }
                }

            }
            best_error = calculateTotalError(input , output);

            System.out.println("Best error="+best_error+"\t\tepoch="+epoch);

            //reduce alpha, if learning is stagnant
            if(Math.random() <= update_probability && (Math.abs(previous_best_error-best_error) < Math.pow(10,-7))){alpha = alpha/10;}
            if(best_error == 0){
                System.out.println("NeuralNetwork.java : [Best accuracy achieved. Training stopped.]");
                break;
            }
            if(alpha == 0){
                System.out.println("NeuralNetwork.java : [Maximum accuracy achieved. Training stopped.]");
                break;
            }
            // print_weights();
            // print_biases();
        }

    }

    //clean display of weights
    public void print_weights(){
        System.out.println("\n----------------------------------------");
        System.out.println("\tWEIGHTS:\n");
        for (int i = 0; i < structure.length - 1; i++) {
            System.out.println("Weight matrix from layer "+(i) + " to layer "+(i+1));
            weights[i].print_matrix();
            System.out.println();
        }
        System.out.println("----------------------------------------\n");
    }

    //clean display of biases
    public void print_biases(){
        System.out.println("\n----------------------------------------");
        System.out.println("\tBIASES:\n");
        for (int i = 0; i < structure.length - 1; i++) {
            System.out.println("Bias matrix from layer "+(i) + " to layer "+(i+1));
            biases[i].print_matrix();
            System.out.println();
        }
        System.out.println("----------------------------------------\n");
    }

    /**
    Simplified and readable implementation of Genetic Algorithm. 
    codes below this can potentially work.
    deletion of all codes below will not affect the other codes of the library.
     */

    public void train_GA_v1(double[][] inputs, double[][] targets, int generations) {
        int populationSize = 10;
        int numChanges = 10;

        // Step 1: Initialize a population of neural networks with random weights
        NeuralNetwork[] population = new NeuralNetwork[populationSize];
        for (int i = 0; i < populationSize; i++) {
            population[i] = new NeuralNetwork(structure, true); // Assuming you want to initialize randomly
        }

        for (int generation = 0; generation < generations; generation++) {
            System.out.println("gen " + generation + " : " + Arrays.toString(population));

            // Step 2: Make small changes in weights for each network
            for (int i = 0; i < populationSize; i++) {
                for (int j = 0; j < numChanges; j++) {
                    mutateWeights(population[i], 0.1); // You can adjust the mutation rate as needed
                    double mutatedFitness = 1.0 / (1.0 + population[i].calculateTotalError(inputs, targets));

                    // No need to create new objects, directly modify weights and biases
                    // Choose the better one between the original and the mutated network
                    if (mutatedFitness > 1.0 / (1.0 + population[i].calculateTotalError(inputs, targets))) {
                        // The weights and biases are already modified in the original network
                        // as mutateWeights directly modifies the object
                    }
                }
            }

            // Step 3: Choose the best network
            double[] fitness = new double[populationSize];
            for (int i = 0; i < populationSize; i++) {
                fitness[i] = 1.0 / (1.0 + population[i].calculateTotalError(inputs, targets));
            }

            int bestIndex = selectIndividual(fitness); // You can reuse the selectIndividual method

            // Step 4: Populate the array with the chosen best network
            for (int i = 0; i < populationSize; i++) {
                copyWeights(population[i], population[bestIndex]);
            }
        }
    }

    // Helper method to copy weights directly
    private void copyWeights(NeuralNetwork destination, NeuralNetwork source) {
        for (int layer = 0; layer < destination.weights.length; layer++) {
            for (int i = 0; i < destination.weights[layer].rows; i++) {
                for (int j = 0; j < destination.weights[layer].columns; j++) {
                    destination.weights[layer].set(i, j, source.weights[layer].get(i, j));
                }
            }
        }

        // Similar copying can be applied to biases if needed
    }

    // Helper method to mutate weights directly
    private void mutateWeights(NeuralNetwork network, double mutationRate) {
        Random random = new Random();
        for (int layer = 0; layer < network.weights.length; layer++) {
            for (int i = 0; i < network.weights[layer].rows; i++) {
                for (int j = 0; j < network.weights[layer].columns; j++) {
                    if (random.nextDouble() < mutationRate) {
                        // Apply a small random change to the weight
                        network.weights[layer].set(i, j, network.weights[layer].get(i, j) + random.nextGaussian());
                    }
                }
            }
        }

        // Similar mutation can be applied to biases if needed
    }

    public void train_GA_v2(double[][] inputs, double[][] targets, int populationSize, int generations) {
        int crossover_k = 2;
        double mutation_p = 0.1;

        // Create an initial population of neural networks
        NeuralNetwork[] population = new NeuralNetwork[populationSize];
        for (int i = 0; i < populationSize; i++) {
            population[i] = new NeuralNetwork(structure, true); // Assuming you want to initialize randomly
        }

        for (int generation = 0; generation < generations; generation++) {

            // Evaluate the fitness of each individual in the population
            double[] fitness = new double[populationSize];
            for (int i = 0; i < populationSize; i++) {
                fitness[i] = 1.0 / (1.0 + population[i].calculateTotalError(inputs, targets));
            }

            // Select the top individuals based on fitness
            NeuralNetwork[] selected = new NeuralNetwork[populationSize];
            int index = selectIndividual(fitness);
            for (int i = 0; i < populationSize; i++) {

                // Debugging statement
                System.out.println("Selected index for parent " + i + ": " + index);

                copyNet(selected[i], population[index]); // Assuming you have a clone method in NeuralNetwork
            }

            // Apply crossover and mutation to create a new generation
            for (int i = 0; i < populationSize; i += 2) {
                crossover(selected[i], selected[i + 1], crossover_k);
                mutate(selected[i], mutation_p);
                mutate(selected[i + 1], mutation_p);
            }

            // Replace the old population with the new generation
            population = Arrays.copyOf(selected, populationSize);

        }
    }
    // Helper function to select an individual based on fitness
    private int selectIndividual(double[] fitness) {
        double totalFitness = Arrays.stream(fitness).sum();
        double randomValue = random.nextDouble() * totalFitness;

        double sum = 0;
        for (int i = 0; i < fitness.length; i++) {
            sum += fitness[i];
            if (sum >= randomValue) {
                int selected = Math.min(i, fitness.length - 1);

                // Debugging statement
                System.out.println("Selected index: " + selected);

                return selected;
            }
        }

        // This should not happen, but just in case
        return fitness.length - 1;
    }

    // Helper function to perform crossover between two neural networks
    private void crossover(NeuralNetwork parent1, NeuralNetwork parent2, int kPoints) {
        // Ensure that both parents have the same structure
        if (!Arrays.equals(parent1.structure, parent2.structure)) {
            System.out.println("Incompatible parents for crossover.");
            return;
        }

        // Debugging statement
        System.out.println("Crossover between parents: " + Arrays.toString(parent1.structure));

        // Ensure that both parents have the same structure
        if (!Arrays.equals(parent1.structure, parent2.structure)) {
            System.out.println("Incompatible parents for crossover.");
            return;
        }

        // Randomly select k crossover points
        int[] crossoverPoints = new int[kPoints];
        for (int i = 0; i < kPoints; i++) {
            crossoverPoints[i] = random.nextInt(parent1.weights.length * 2); // Considering both weights and biases
        }

        // Sort the crossover points for easier processing
        Arrays.sort(crossoverPoints);

        int currentPoint = 0;
        boolean swap = false;

        // Iterate through the layers of the neural network
        for (int i = 0; i < parent1.weights.length; i++) {
            // Check for null matrices in parents
            if (parent1.weights[i] == null || parent1.biases[i] == null ||
            parent2.weights[i] == null || parent2.biases[i] == null) {
                System.out.println("Null matrix in crossover.");
                return;
            }

            Matrix temp;

            // Swap genetic material between parents at crossover points
            if (swap) {
                // Swap weights
                temp = parent1.weights[i];
                parent1.weights[i] = parent2.weights[i];
                parent2.weights[i] = temp;

                // Swap biases
                temp = parent1.biases[i];
                parent1.biases[i] = parent2.biases[i];
                parent2.biases[i] = temp;
            }

            // Check if the current point is a crossover point
            if (currentPoint < kPoints && i * 2 == crossoverPoints[currentPoint]) {
                swap = !swap; // Toggle swapping behavior at crossover points
                currentPoint++;
            }
        }
    }

    private void mutate(NeuralNetwork individual, double mutationProbability) {
        // Debugging statement
        System.out.println("Mutating individual with structure: " + Arrays.toString(individual.structure));
        // Check for null individual
        if (individual == null) {
            System.out.println("Null individual in mutation.");
            return;
        }

        for (int i = 0; i < individual.weights.length; i++) {
            // Check for null matrices in individual
            if (individual.weights[i] == null || individual.biases[i] == null) {
                System.out.println("Null matrix in mutation.");
                return;
            }

            // Mutate weights
            for (int row = 0; row < individual.weights[i].rows; row++) {
                for (int col = 0; col < individual.weights[i].columns; col++) {
                    if (random.nextDouble() < mutationProbability) {
                        // Mutate with a random value between -1 and 1
                        individual.weights[i].data[row][col] = 2 * random.nextDouble() - 1;
                    }
                }
            }

            // Mutate biases
            for (int row = 0; row < individual.biases[i].rows; row++) {
                if (random.nextDouble() < mutationProbability) {
                    // Mutate with a random value between -1 and 1
                    individual.biases[i].data[row][0] = 2 * random.nextDouble() - 1;
                }
            }
        }
    }
}

    
