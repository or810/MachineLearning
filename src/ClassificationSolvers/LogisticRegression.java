package ClassificationSolvers;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * A class representing the logistic regression method, for binary output.
 * Hence, it is composed of an array of weights, a bias, and
 * a learningRate constant. 
 * 
 * implements the Serlizeable interface for Saving the model.
 */
public class LogisticRegression implements Serializable{

    private double[] weights;
    private double bias;
    private final double learningRate = 0.1;

    /**
     * The only constructor inside the class,
     * that validates the input, and initializes
     * random values for the parameters. 
     * 
     * @param in the number of features the 
     * for the model.
     */
    public LogisticRegression(int in) {
        if(in <= 0)
            throw new RuntimeException("Error! Invalid Dimensions.");

        // an object for generating random numbers.
        Random random = new Random();
        weights = new double[in];
        bias = random.nextDouble();
        for(int i = 0;i<weights.length;i++)
            weights[i] = random.nextDouble();
    }

    /**
     * @param input a vector, composed of the input
     * features of the model.
     * 
     * @return the model's guess based on it's training.
     */
    public double predict(double[] input) {
        return sigmoid(dotProduct(weights, input) + bias);
    }

    private static double dotProduct(double[] a, double[] b) {
        double sum = 0;
        if(a.length != b.length)
            throw new RuntimeException("Error! Invalid Dimensions.");
        for(int i = 0;i<a.length;i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    /**
     * The method calls the overriden 'train' method,
     * for every epoch and dataset size, for trainin,
     * and also shows the progress of training to the user.
     * 
     * @param input the input features for the model.
     * @param target the labeled data of the model.
     * @param epochs the number of trainings done on the dataset.
     */
    public void train(double[][] input, double[] target, int epochs) {
        double presentage = 0;
        for(int epoch = 0;epoch<epochs;epoch++) {
            double lossDerivative = 0;
            double lossDerivativeAbs = 0;
            for(int j = 0;j<target.length;j++) {
                double val = (train(input[j], target[j], input.length));
                lossDerivative += val;
                lossDerivativeAbs += Math.abs(val);
            }

            // metrics:
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
    private double train(double[] input, double target, int length) {
        // calculate initial loss:
        double output = predict(input);
        double sum = 0;
        double loss = (output - target); // loss function.
        sum += loss; // for metrics
        bias -= learningRate * loss;

        for(int j = 0;j<input.length;j++)
            weights[j] -= learningRate * loss
                * (input[j]);
        return sum;
    }

    /**
     * returns the user the result of testing all the given data.
     * 
     * @param input the input features for the model.
     */
    public void testAllData(double[][] input) {
        for(int i = 0;i<input.length;i++) {
            System.out.println((i + 1) + ": " + Arrays.toString(input[i]) + "  ->  " + (predict(input[i])));
        }
    }
}
