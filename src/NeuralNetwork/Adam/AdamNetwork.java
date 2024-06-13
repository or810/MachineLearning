package NeuralNetwork.Adam;

import java.io.Serializable;
import java.util.Arrays;

import Utilities.Activation;
import Utilities.Loss;

/**
 * A class represeting a Neural Network - a multi layered Perceptron
 * capable of handling more complex tasks.
 * It's composed of an array of Dense Layers, the appropriate loss
 * function, the learningRate constant, and each layer's configuration:
 * <pre>
 * their activation function and length.
 * 
 * </pre>
 * implements the Serlizeable interface for saving or loading the model.
 */
public class AdamNetwork implements Serializable{

    /**
     * An array of Dense layers - the main component of the class.
     */
    private AdamLayer[] layers;
    /**
     * An array composed of the appropriate Activation function for
     * each layer index.
     */
    private int[] layerConfig;
    /**
     * An array composed of the number of neurons for each layer index.
     */
    private Activation[] activationsConfig;
    private Loss loss;
    private double learningRate = 0.1;
    private int t;

    /**
     * The only constuctor inside the class.
     * It validates the input, and initailizes
     * random values for the MODEL'S parameters.
     * 
     * The most convienient constructor i thought
     * of for java, as it doesn't support kwargs.
     * 
     * @param parameters only Integers (size of layer)
     * and Activations(layer activation) object
     * instances are accepted. Such that the integer comes
     * first.
     * <pre>
     * 
     * Example: <code>
     * new NeuralNetwork(
     *      3,Activatin.Sigmoid,
     *      1,Activation.Sigmoid
     * );
     * </code></pre>
     * 
     * Therefore, <code> parameters.length </code> must be even.
     */
    public AdamNetwork(Object... parameters) {
        if(parameters.length % 2 == 1)
            throw new RuntimeException("Invalid input for NeuralNetwork.");

        layerConfig = new int[parameters.length / 2];
        activationsConfig = new Activation[parameters.length / 2];
        int count = 0;

        for(int i = 0;i < parameters.length;i++) {
            if(parameters[i] instanceof Integer && (int)parameters[i] >= 0
            && parameters[i + 1] instanceof Activation) {
                layerConfig[count] = (int) parameters[i];
                activationsConfig[count++] = (Activation) parameters[i + 1];
            } else if(! (parameters[i] instanceof Activation))
                throw new RuntimeException("Invalid input for NeuralNetwork.");
        }

        layers = new AdamLayer[layerConfig.length];
        initializeLayers(); 
        combineLayers();
    }

    /**
     * Creates new instances of all the layers.
     */
    private void initializeLayers() {
        layers[layers.length - 1] = new AdamLayer(
            layerConfig[layers.length - 1],
            activationsConfig[layers.length - 1]);
        for(int i = 0;i<layers.length;i++)
            layers[i] = new AdamLayer(layerConfig[i], activationsConfig[i]);
    }

    /**
     * Connects BinNode-wise each layer to it's appropriate
     * neighbors such that the edge's only have one neighbor
     * and the other is null.
     */
    public void combineLayers() {
        layers[0].setNeighbors(null, layers[1]);
        layers[layers.length -1].setNeighbors(layers[layers.length - 2], null);

        for(int i = 1;i<layers.length - 1;i++)
            layers[i].setNeighbors(layers[i - 1], layers[i + 1]);
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
    public void train(double[][] inputs, double[][] targets, int epochs) {
        double presentage = 0;
        for(int epoch = 0;epoch<epochs;epoch++) {
            double lossDerivative = 0;
            double lossDerivativeAbs = 0;
            for(int j = 0;j<targets.length;j++) {
                double val = (train(inputs[j], targets[j], inputs.length));
                lossDerivative += val;
                lossDerivativeAbs += Math.abs(val);
            }

            double progress = ((double) (epoch + 1) / epochs) * 100;
            if(progress >= presentage + 1.0) {
                presentage = (int) progress;
                String message = String.format("Epoch: %-15s Loss: (slope:  %-5.8e,   abs: %-1.8e)        Progress: %.0f%%"
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
    private double train(double[] input, double[] target, int length) {
        double[] output = predict(input);
        double[] initialCost = loss.calcLoss(input, target, output);
        double sum = 0;
        for(int i = 0;i<initialCost.length;i++) {
            initialCost[i] /= length;
            sum += initialCost[i];
        }
        layers[layers.length - 2].backpropagation(
            initialCost, this);


        return sum;
    }


    /**
     * @param input a vector, composed of the input
     * features of the model.
     * 
     * @return the model's guess based on it's training.
     */
    public double[] predict(double[] input) {

        layers[0].setCachedActivations(input);
        input = layers[1].feedForward(input);

        return input;
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

    
    public Loss getLoss() {
        return loss;
    }
        
    public void setLoss(Loss loss) {
        this.loss = loss;
    }
            
    public double getLearningRate() {
        return learningRate;
    }
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getT() {
        return t;
    }

    public void incrementT() {
        t++;
    }
    
}
