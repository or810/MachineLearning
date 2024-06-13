package ClassificationSolvers;

import LinearRegression.LinearRegression;
import Utilities.Model;

public class LogisticRegressionTest {

    static double[][] input = {{1,0,0}, {0,0,0}, {1,1,1}, {0,1,0}};
    static double[] output = {1,0,0,1};

    public static void main(String[] args) {
        train();
        
    }

    public static void train() {
        LogisticRegression logisticRegression = new LogisticRegression(3);
        
        
        // linearRegression.testAllData(input, output);
        logisticRegression.train(input, output, 10000);
        logisticRegression.testAllData(input);
        System.out.println("Successfully used Logistic Regression!");

    }

    public static void load() {
        LogisticRegression logisticRegression = (LogisticRegression)Model.load("logisticRegressionModel");

        logisticRegression.testAllData(input);

        System.out.println("Successfully loaded a Logistic regression model!");


    }
}
