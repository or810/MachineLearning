package NeuralNetwork.Adam;

import NeuralNetwork.Normal.NeuralNetwork;
import Utilities.Activation;
import Utilities.Loss;
import Utilities.Model;

public class AdamTest {
    // //Classification of XOR gate with feature Engineering.
    static double[][] inputC = {{1,0,0}, {0,0,0}, {1,1,1}, {0,1,0}};
    static double[][] outputC = {{1},{0},{0},{1}};

    // //Linear Regression with irrelevant data.
    static double[][] inputR = {{1,20,50},{2,20,50},{3,20,50},{4,20,50},{6,20,50}};
    static double[][] outputR = {{4,8},{8,12},{12,16},{16,20},{24,28}};
    public static void main(String[] args) {
        
        // trainClassify();
        // loadClassify();

        trainRegression();
        // loadRegression();

    }

    private static void trainClassify() {
        AdamNetwork network = new AdamNetwork(
            3, Activation.Sigmoid,
            4, Activation.LeakyRelu,
            1, Activation.Sigmoid
        );

        network.setLearningRate(0.3);
        network.setLoss(Loss.BinaryCrossEntropy);

        network.train(inputC, outputC, 10000);
        network.testAllData(inputC);
        Model.save(network, "AdamNetworkClassify");

        System.out.println("Successfully trained an Adam Neural Network for Classification!");
    }

    private static void loadClassify() {

        AdamNetwork network = (AdamNetwork)Model.load("AdamNetworkClassify");

        network.testAllData(inputC);

        System.out.println("Successfully loaded an Adam NeuralNetwork for Classification!");
    }


    private static void trainRegression() {
        AdamNetwork network = new AdamNetwork(
            3, Activation.TANH,
            5, Activation.LeakyRelu,
            2, Activation.Relu
        );

        network.setLearningRate(0.003);
        network.setLoss(Loss.MSE);

        network.train(inputR, outputR, 10000);
        network.testAllData(inputR);

        Model.save(network, "AdamNetworkRegression");
        System.out.println("Successfully trained an Adam Neural Network for Regression!");
    }

    private static void loadRegression() {

        AdamNetwork network = (AdamNetwork)Model.load("AdamNetworkRegression");

        network.testAllData(inputR);

        System.out.println("Successfully loaded an Adam Neural Network for Regression!");
    }
}
