package Utilities;

import java.io.Serializable;
/*
 * A Class of Constants for the initial loss/cost functions,
 * that calculated the appropriate loss function's derivative.
 * 
 * *note that throughout the entire project i may use the words
 * loss/cost/errors interchangably.
 */
public enum Loss{
    // types of cost/loss functions.
    BinaryCrossEntropy,
    MSE;

    public double[] calcLoss(double[] input, double[] target, double[] output) {
        double[] errors = new double[target.length];
        for(int i = 0;i<errors.length;i++) {
            errors[i] = switch (this) {
                case BinaryCrossEntropy -> (output[i] - target[i]);
                case MSE -> (output[i] - target[i]) * input[i];
            };
        }
        return errors;
    }
}

