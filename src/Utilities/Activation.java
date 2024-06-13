package Utilities;

import java.io.Serializable;

/**
 * A class of Activation funcion constants, that deals with 
 * calculating the appropriate activation function, even for
 * their derivative ones.
 */
public enum Activation implements Serializable {

    // types of activations:
    Linear,
    LeakyRelu,
    Relu,
    TANH,
    Sigmoid;

    public double apply(double x) {
        return switch (this) {
            case Sigmoid -> 1 / (1 + Math.exp(-x));
            case Relu -> Math.max(0, x);
            case LeakyRelu -> Math.max(0.01 * x, x);
            case TANH -> Math.tanh(x);
            case Linear -> x;
        };
    }


    public double applyDerivative(double x) {
        return switch (this) {
            case Linear -> 1;
            case Relu -> (x > 0) ? 1 : 0;
            case LeakyRelu -> (x > 0) ? 0.01 : 1;
            
            // This part is mathematically incorrect though,
            // call it after calling 'apply':
            case TANH -> 1 - (x * x);
            case Sigmoid -> x * (1 - x);
        };
    }
}