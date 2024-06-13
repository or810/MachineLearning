package NeuralNetwork.Adam;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

import Utilities.Activation;

/**
 * A Dense layer class with a BinNode-like structure for the core
 * functionality of the model to stay here.
 * Likewise, it's also
 * composed of a double matrix for weights, a double bias,
 * the number of neurons - length, the cached-activations - the
 * last saved activations (post activation function), and
 * the activation function. 
 * <pre>
 * </pre>
 * Furthermore, this class was idealy designed to only be
 * constructed and used from the 'NeuralNetwork' class. 
 * <pre>
 * </pre>
 * implements the Serlizeable interface for Saving the model.
 */
class AdamLayer implements Serializable{
    /*
     * BinNode Structure:
     */
    private AdamLayer next;
    private AdamLayer prev;

    private double[][] weights;
    private double[][] m;
    private double[][] v;
    private double bias;
    private int length;

    /**
     * Additional Constants for Adam:
     */
    private static final double beta1 = 0.9;
    private static final double beta2 = 0.99;
    private static final double epsilon = 1e-14;

    /**
     * The last saved activation after the
     * appropriate activation function was
     * applied.
     */
    private double[] cachedActivations;

    private Activation func;

    /**
     * The only Constructor in the class.
     * Doesn't handle errors as it assumes
     * it all came well from NeuralNetwork class.
     * 
     * @param length the number of Neurons in the layer.
     * @param func the layer's activation function.
     */
    public AdamLayer(int length, Activation func) {
        this.func = func;
        this.length = length;
        cachedActivations = new double[length];
    }

    /**
     * Sets the neighbors of the layer.
     * 
     * @param prev the previous layer.
     * @param next the next layer
     */
    public void setNeighbors(AdamLayer prev, AdamLayer next) {
        this.prev = prev;
        if(next != null) {
            this.next = next;
            weights = new double[next.length][length];
            m = new double[next.length][length];
            v = new double[next.length][length];
            randomize();
        }
    }

    /**
     * Randomizes the weights and biases,
     * using the random object instance of
     * the Random class.
     */
    public void randomize() {
        Random random = new Random();
        bias = random.nextDouble();
        for(int i = 0;i<weights.length;i++)
            for(int j = 0;j<weights[0].length;j++)
                weights[i][j] = random.nextDouble();
    }

    /**
     * The function was designed to work for every layer
     * but the first one, as it calls the 'prev' BinNode.
     * though, the first one is trivially handled in 
     * the NeuralNetwork class.
     * 
     * @param input the input from the previous layer,
     * of the previous layer's size.
     * @return the output from this layer, of this layer's size.
     */
    public double[] feedForward(double[] input) {
        for(int i = 0;i<length;i++) {
            double z = dotProduct(input, prev.weights[i]) + bias;
            cachedActivations[i] = func.apply(z);
        }
        // if this isn't the last layer, keep the feedForward.
        return ((next != null) ? next.feedForward(cachedActivations)
                               : cachedActivations);
    }


    /**
     * Performs the backpropagation algorithm, that implements
     * it's simplified formulas.
     * 
     * @param prevError delta of the NEXT layer. Though It's still
     * counts it as the previous loss calculated - thus the name. 
     * @param network a reference for the previous layer - barely used
     * but it was chosen instead of potentially more method inputs
     * in case of furture additions.
     */
    public void backpropagation(double[] prevError, AdamNetwork network) {

        //2:
        double[] errors = new double[length];
        for(int i = 0;i<length;i++) { // for each neuron in this layer.
            double errorSum = 0;
            for(int j = 0;j< next.length;j++) { // for each neuron in next layer.
                errorSum += prevError[j] * weights[j][i];
            }
            // calculate the error for this layer.
            errors[i] = errorSum * func.applyDerivative(cachedActivations[i]);
        }
        //3:
        network.incrementT();// does t++
        for(int i = 0;i<next.length;i++) {
            for(int j = 0;j < length;j++) {

                // for each weight:
                //1:
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * prevError[i] * cachedActivations[j];
                //2:
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * Math.pow(prevError[i] * cachedActivations[j], 2);
                //3:
                double mHat = m[i][j] / (1 - Math.pow(beta1, network.getT()));
                double vHat = v[i][j] / (1 - Math.pow(beta2, network.getT()));


                weights[i][j] -= network.getLearningRate() * prevError[i] * cachedActivations[j];
            }
        }

        double deltabiasSum = 0;
        for(int i = 0;i<length;i++)
            deltabiasSum += errors[i];
        // 4:
        bias += network.getLearningRate() * deltabiasSum / length;

        // if this isn't the last layer, propagate.
        if(prev != null)
            prev.backpropagation(errors, network);

    }


    private static double dotProduct(double[] a, double[] b) {
        double sum = 0;
        if(a.length != b.length)
        throw new RuntimeException("Error! Invalid Dimensions.");
        for(int i = 0;i<a.length;i++)
            sum += a[i] * b[i];
        return sum;
    }

    public void setCachedActivations(double[] cachedActivations) {
        this.cachedActivations = cachedActivations;
    }


    public double[] getCachedActivations() {
        return cachedActivations;
    }  
}
