package NeuralNetwork.Normal;


import NeuralNetwork.Adam.AdamNetwork;
import Utilities.Activation;
import Utilities.Loss;
import Utilities.Model;

public class NeuralNetworkTest {

    // //Classification of XOR gate with feature Engineering.
    static double[][] inputC = {{1,0,0}, {0,0,0}, {1,1,1}, {0,1,0}};
    static double[][] outputC = {{1},{0},{0},{1}};

    // //Linear Regression with irrelevant data.
    static double[][] inputR = {{1,20,50},{2,20,50},{3,20,50},{4,20,50},{6,20,50}};
    static double[][] outputR = {{4,8},{8,12},{12,16},{16,20},{24,28}};
    public static void main(String[] args) {
        
        trainClassify();
        // loadClassify();

        // trainRegression();
        // loadRegression();

    }

    private static void trainClassify() {
        NeuralNetwork network = new NeuralNetwork(
            3, Activation.Sigmoid,
            4, Activation.TANH,
            1, Activation.Relu
        );

        network.setLearningRate(0.08);
        network.setLoss(Loss.BinaryCrossEntropy);

        network.train(inputC, outputC, 10000);
        network.testAllData(inputC);
        Model.save(network, "NeuralNetworkClassify");

        System.out.println("Successfully trained a Neural Network for Classification!");
    }

    private static void loadClassify() {

        NeuralNetwork network = (NeuralNetwork)Model.load("NeuralNetworkClassify");

        network.testAllData(inputC);

        System.out.println("Successfully loaded a NeuralNetwork for Classification!");
    }


    private static void trainRegression() {
        NeuralNetwork network = new NeuralNetwork(
            3, Activation.TANH,
            4, Activation.LeakyRelu,
            2, Activation.Relu
        );

        network.setLearningRate(0.0003);
        network.setLoss(Loss.MSE);

        network.train(inputR, outputR, 10000);
        network.testAllData(inputR);

        // Model.save(network, "NeuralNetworkRegression");
        System.out.println("Successfully trained a Neural Network for Regression!");
    }

    private static void loadRegression() {

        NeuralNetwork network = (NeuralNetwork)Model.load("NeuralNetworkRegression");

        network.testAllData(inputR);

        System.out.println("Successfully loaded a Neural Network for Regression!");
    }
    
    
}
