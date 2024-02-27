import java.util.*;
import java.util.function.*;
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
    
    //for random initialization
    double lowest_weight = -32;
    double highest_weight = 32;
    
    //for random initialization
    double lowest_bias = -8;
    double highest_bias = 8;

    boolean update_bias = false;//bias will not be updated during training if falsw

    boolean initialize_bias = false;//all bias will be 0 if false

    String activation = "sigmoid";//choose between "sigmoid", "tanh", "relu"

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
                        //store old weight
                        double weight = weights[i].data[row][col];

                        if(Math.random() < update_probability){

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
                        else{
                            //set weights to a random offset
                            if(Math.random() < update_probability){weights[i].data[row][col] = weight + alpha;}
                            else{weights[i].data[row][col] = weight - alpha;}
                            best_error = Math.abs(calculateTotalError(input , output));
                        }
                    }
                }
                if(update_bias){
                    for (int row = 0; row < biases[i].rows; row++) {
                        //store old bias
                        double bias = biases[i].data[row][0];
                        if(Math.random() < update_probability){

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
                        else{
                            //set biases to a random offset
                            if(Math.random() < update_probability){biases[i].data[row][0] = bias + alpha;}
                            else{biases[i].data[row][0] = bias - alpha;}
                            best_error = Math.abs(calculateTotalError(input , output));
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
}

class Matrix {
    public int rows;
    public int columns;
    public double[][] data;

    /**
    Simple Matrix library
    Coded by Nihal Gazi for Azinet
     */

    // Constructor to create a matrix with given rows and columns
    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.data = new double[rows][columns];
    }

    public static Matrix from_array(double[] arr) {
        Matrix result = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            result.data[i][0] = arr[i];
        }
        return result;
    }

    public Matrix(String input) {
        String[] rowsArray = input.split(" ; "); // Split input into rows
        this.rows = rowsArray.length;
        this.columns = rowsArray[0].split(" ").length;
        this.data = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            String[] values = rowsArray[i].split(" "); // Split row into individual values
            for (int j = 0; j < columns; j++) {
                this.data[i][j] = Double.parseDouble(values[j]);
            }
        }
    }

    public Matrix map(Function<Double, Double> function) {
        Matrix result = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = function.apply(data[i][j]);
            }
        }
        return result;
    }

    public void fill(double val){
        for(int i=0;i<data.length;i++){
            for(int j=0;j<data[0].length;j++){
                data[i][j]=val;
            }
        }
    }

    // Method to set the value at a specific row and column
    public void set(int row, int column, double value) {
        if (row >= 0 && row < rows && column >= 0 && column < columns) {
            data[row][column] = value;
        } else {
            System.out.println("Invalid row or column index.");
        }
    }

    public void set(String input) {
        String[] rowsArray = input.split(" ; "); // Split input into rows
        this.rows = rowsArray.length;
        this.columns = rowsArray[0].split(" ").length;
        this.data = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            String[] values = rowsArray[i].split(" "); // Split row into individual values
            for (int j = 0; j < columns; j++) {
                this.data[i][j] = Double.parseDouble(values[j]);
            }
        }
    }

    // Method to get the value at a specific row and column
    public double get(int row, int column) {
        if (row >= 0 && row < rows && column >= 0 && column < columns) {
            return data[row][column];
        } else {
            System.out.println("Invalid row or column index.");
            return -1; // You can choose an appropriate default value
        }
    }

    // Method to print the matrix
    public void print_matrix() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                System.out.print(data[i][j] + "\t");
            }
            System.out.println();
        }
    }

    public static Matrix multiply(Matrix matrix1, Matrix matrix2) {
        int rowsA = matrix1.rows;
        int columnsA = matrix1.columns;
        int rowsB = matrix2.rows;
        int columnsB = matrix2.columns;

        // Check if the matrices can be multiplied
        if (columnsA != rowsB) {
            System.out.println("Cannot multiply the matrices. Invalid dimensions.");
            System.out.println("dim1="+rowsA+"x"+columnsA);
            System.out.println("dim2="+rowsB+"x"+columnsB);
            System.exit(0);
            return null;
        }

        Matrix result = new Matrix(rowsA, columnsB);

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < columnsB; j++) {
                double sum = 0;
                for (int k = 0; k < columnsA; k++) {
                    sum += matrix1.data[i][k] * matrix2.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    public Matrix multiply(double scalar) {
        Matrix result = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = this.data[i][j] * scalar;
            }
        }
        return result;
    }

    public static Matrix transpose(Matrix inputMatrix) {
        int rows = inputMatrix.columns;
        int columns = inputMatrix.rows;
        Matrix result = new Matrix(rows, columns);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = inputMatrix.data[j][i];
            }
        }

        return result;
    }

    public static Matrix add(Matrix matrix1, Matrix matrix2) {
        int rowsA = matrix1.rows;
        int columnsA = matrix1.columns;
        int rowsB = matrix2.rows;
        int columnsB = matrix2.columns;

        // Check if the matrices can be added
        if (rowsA != rowsB || columnsA != columnsB) {
            System.out.println("Cannot add the matrices. Invalid dimensions.");
            return null;
        }

        Matrix result = new Matrix(rowsA, columnsA);

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < columnsA; j++) {
                result.data[i][j] = matrix1.data[i][j] + matrix2.data[i][j];
            }
        }

        return result;
    }

    public static Matrix subtract(Matrix matrix1, Matrix matrix2) {
        int rowsA = matrix1.rows;
        int columnsA = matrix1.columns;
        int rowsB = matrix2.rows;
        int columnsB = matrix2.columns;

        // Check if the matrices can be subtracted
        if (rowsA != rowsB || columnsA != columnsB) {
            System.out.println("Cannot add the matrices. Invalid dimensions.");
            return null;
        }

        Matrix result = new Matrix(rowsA, columnsA);

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < columnsA; j++) {
                result.data[i][j] = matrix1.data[i][j] - matrix2.data[i][j];
            }
        }

        return result;
    }

    //for neural network
    public double sumOfSquares() {
        double sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                sum += data[i][j] * data[i][j];
            }
        }
        return sum;
    }

}
