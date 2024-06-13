package LinearRegression;

import Utilities.Model;

public class LinearRegressionTest {
    // training with some irrelevant input features.
    static double[][] input = {{1,2,5},{2,2,5},{3,2,5},{4,2,5},{6,2,5}};
    static double[][] output = {{4,8},{8,12},{12,16},{16,20},{24,28}};
    public static void main(String[] args) {
        

        
    }
    
    public static void train() {
        LinearRegression linearRegression = new LinearRegression(3, 2);
        
        
        // linearRegression.testAllData(input, output);
        linearRegression.train(input, output, 10000);
        linearRegression.testAllData(input, output);
        System.out.println("Successfully used Linear Regression!");

    }

    public static void load() {
        LinearRegression linearRegression = (LinearRegression)Model.load("LinearRegression");

        linearRegression.testAllData(input, output);

        System.out.println("Successfully loaded a Linear regression model!");


    }

}
