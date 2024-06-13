package Perceptron;


import Utilities.Activation;
import Utilities.Model;

public class PerceptronTest {

    //Classification with feature Engineering.
    static double[][] inputC = {{1,0,0}, {0,0,0}, {1,1,1}, {0,1,0}};
    static double[][] outputC = {{1},{0},{0},{1}};

    //Regression with irrelevant data.
    static double[][] inputR = {{1,2,5},{2,2,5},{3,2,5},{4,2,5},{6,2,5}};
    static double[][] outputR = {{4,8},{8,12},{12,16},{16,20},{24,28}};
    
    public static void main(String[] args) {
        // trainRegression();
        // loadClassify();
        // trainClassify();
        // loadClassify();
    }

    public static void trainRegression() {
        Perceptron perceptron = new Perceptron(3, 2, Activation.LeakyRelu);
        
        
        perceptron.train(inputR, outputR, 10000);
        perceptron.testAllData(inputR);
        System.out.println("Successfully used a Perceptron for Regression!");

        Model.save(perceptron, "PerceptronRegression");

    }

    public static void loadRegression() {
        Perceptron perceptron = (Perceptron)Model.load("PerceptronRegression");

        perceptron.testAllData(inputR);

        System.out.println("Successfully loaded a Perceptron model for Regression!");


    }

    public static void trainClassify() {
        Perceptron perceptron = new Perceptron(3, 1, Activation.TANH);
        
        
        // linearRegression.testAllData(input, output);
        perceptron.train(inputC, outputC, 100000);
        perceptron.testAllData(inputC);
        System.out.println("Successfully used a Perceptron for Classification!");

        Model.save(perceptron, "PerceptronClassify");
    }

    public static void loadClassify() {
        Perceptron linearRegression = (Perceptron)Model.load("PerceptronClassify");

        linearRegression.testAllData(inputC);

        System.out.println("Successfully loaded a Perceptron model for Classification!");


    }
}
