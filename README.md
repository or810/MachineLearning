# MachineLearning
An implementation From Scratch of NeuralNetworks in Java.

### Done one step at a time:

### LinearRegression, LogisticRegression =>

### => Perceptron =>

### => NeuralNetwork(MLP) =>

### => NeuralNetwork with adam.


## General API:

### Model Saving And Loading:
```java
public static void save(Serlizeable object, String filename);

public static Object load(String filename);
// Note:
// As per Polymophism rules, the given object
// should be downcasted explicitly.
// Moreover, only models from this project
// may be loaded.
```

### The model's prediction based on what it has learned:
```java
public double[] predict(double[] input);
```

### The model's predictions for the whole dataset:
```java
public void testAll(double[][] input);
```

### For any type except Logistic Regression:
```java
public void train(double[][] input, double[][] target, int epochs);
```
### For Logistic Regression:
```java
 public void train(double[][] input, double[] target, int epochs);
```



