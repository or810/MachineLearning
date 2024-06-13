package Perceptron;
import java.io.Serializable;
import java.util.Arrays;

import Utilities.Activation;

/**
 * A Class represending a Perceptron - a method that can handle
 * both logistic regression and linear regression combined.
 * It's composed of a double matrix for weights, a double 
 * bias, a constant 'learningRate', and an activation function.
 * <pre>
 * </pre>
 * implements the Serlizeable interface for Saving the model.
 */
public class Perceptron implements Serializable{
    private Activation func;
    
    /**
     * The major parameter of the model, designed
     * as a matrix to handle more complex problems,
     * such that it's design follows int's mathematical
     * conventions that:
     * <pre> <code>
     * @param weights = new double[out][in];
     * </code> </pre>
     */
    private double[][] weights;
    private double bias;
    private double learningRate = 0.1;

    /**
     * The only constructor inside the class, that
     * validates the input, and initializes
     *  random
     * values for the parameters.
     * @param in
     * @param out
     * @param func
     */
    public Perceptron(int in, int out, Activation func) {
        if(in < 1 || out < 1)
            throw new RuntimeException("Error! Invalid Dimension.");
        this.func = func;
        weights = new double[out][in];
        bias = Math.random();
        for(int i = 0;i<weights.length;i++)
            for(int j = 0;j<weights[0].length;j++)
                weights[i][j] = Math.random();
    }

    /**
     * @param input a vector, composed of the input
     * features of the model.
     * 
     * @return the model's guess based on it's training.
     */
    public double[] predict(double[] input) {
        double[] output = new double[weights.length];
        for(int i = 0;i<weights.length;i++)
            output[i] = func.apply(dotProduct(weights[i], input) + bias);
        return output;
    }

    private static double dotProduct(double[] a, double[] b) {
        double sum = 0;
        if(a.length != b.length)
            throw new RuntimeException("Error! Invalid Dimensions!");
        for(int i = 0;i<a.length;i++)
            sum += a[i] * b[i];
        return sum;
    }

    /**
     * The method called the overriding 'train' method,
     * for every epoch and dataset size, for training,
     * and also shows the progress of training to the user.
     * 
     * @param input the input features for the model.
     * @param target the labeled data of the model.
     * @param epochs the number of trainings done on the dataset.
     */
    public void train(double[][] input, double[][] target, int epochs) {
        double presentage = 0;
        for(int epoch = 0;epoch<epochs;epoch++) {
            double lossDerivative = 0;
            double lossDerivativeAbs = 0;
            for(int j = 0;j<target.length;j++) {
                double val = (train(input[j], target[j], input.length));
                lossDerivative += val;
                lossDerivativeAbs += Math.abs(val);
            }

            double progress = ((double) (epoch + 1) / epochs) * 100;
            if(progress >= presentage + 1.0) {
                presentage = (int) progress;
                String message = String.format("Epoch: %-15s Loss: (slope:  %-5.3e,   abs: %-1.3e)        Progress: %.0f%%"
                        , epoch + 1, lossDerivative, lossDerivativeAbs, presentage);
                System.out.println(message);
            }
        }
    }

    /**
     * a method that overrides the other 'train' method,
     * it is refactored for an easier reading.
     * 
     * @param input the input features for the model.
     * @param target the labeled output for the model.
     * @param length the size of the dataset from the
     * other 'train method' - used to divide the loss.
     * @return returns the sum of the loss, for metrics.
     */
    public double train(double[] input, double[] target, int length) {
        // calculate initial loss:
        double[] loss = new double[target.length];
        double[] output = predict(input);
        double sum = 0;
        for(int i = 0;i < loss.length;i++) {
            loss[i] = (output[i] - target[i]) / length;
            sum += loss[i];
            bias -= learningRate * loss[i];

            for(int j = 0;j<input.length;j++)
                weights[i][j] -= learningRate * loss[i]
                    * input[j];
        }
        return sum;
    }

    /**
     * returns the user the result of testing all the given data.
     * 
     * @param input the input features for the model.
     * @param target the labeled outputs for the model.
     */
    public void testAllData(double[][] input, double[][] target) {
        for(int i = 0;i<input.length;i++) {
            System.out.println((i + 1) + ": " + Arrays.toString(input[i]) + "  ->  " + Arrays.toString(predict(input[i])));
        }
    }
    
}
