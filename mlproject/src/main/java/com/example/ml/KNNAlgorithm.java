package com.example.ml;

import java.util.List;
import java.util.PriorityQueue;
import java.util.Comparator;

public class KNNAlgorithm {
    private int k;

    public KNNAlgorithm(int k) {
        this.k = k;
    }

    public static class DataPoint {
        public double x;
        public double y;
        public double value;

        public DataPoint(double x, double y, double value) {
            this.x = x;
            this.y = y;
            this.value = value;
        }
    }

    public double predict(List<DataPoint> data, DataPoint target) {
        PriorityQueue<DataPoint> pq = new PriorityQueue<>(k, Comparator.comparingDouble(p -> distance(p, target)));

        for (DataPoint point : data) {
            pq.offer(point);
            if (pq.size() > k) {
                pq.poll();
            }
        }

        double sum = 0;
        for (DataPoint point : pq) {
            sum += point.value;
        }

        return sum / k; //calcular el promedio de los k puntos más cercanos
    }

    private double distance(DataPoint p1, DataPoint p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }
}
