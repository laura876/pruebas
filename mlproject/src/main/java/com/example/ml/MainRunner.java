package com.example.ml;

import java.util.Arrays;
import java.util.List;

public class MainRunner {
    public static void main(String[] args) {
        List<KNNAlgorithm.DataPoint> data = Arrays.asList(
            new KNNAlgorithm.DataPoint(1, 1, 10),
            new KNNAlgorithm.DataPoint(2, 1, 20),
            new KNNAlgorithm.DataPoint(3, 2, 30),
            new KNNAlgorithm.DataPoint(4, 2, 40),
            new KNNAlgorithm.DataPoint(5, 3, 50)
        );

        KNNAlgorithm knn = new KNNAlgorithm(3);
        KNNAlgorithm.DataPoint target = new KNNAlgorithm.DataPoint(3, 3, 0);
        double prediction = knn.predict(data, target);

        System.out.println("Predicted value: " + prediction);
    }
}