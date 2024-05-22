package com.example.ml;

import static org.junit.Assert.assertEquals;
import org.junit.Test;
import java.util.Arrays;
import java.util.List;

public class KNNAlgorithmTest {

    @Test
    public void testPredict() {
        // Configurar datos de entrenamiento
        List<KNNAlgorithm.DataPoint> data = Arrays.asList(
            new KNNAlgorithm.DataPoint(1, 1, 2),
            new KNNAlgorithm.DataPoint(2, 2, 3),
            new KNNAlgorithm.DataPoint(3, 3, 4),
            new KNNAlgorithm.DataPoint(4, 4, 5)
        );

        // Crear instancia del algoritmo KNN con K=2
        KNNAlgorithm knn = new KNNAlgorithm(2);

        // Configurar el punto de destino
        KNNAlgorithm.DataPoint target = new KNNAlgorithm.DataPoint(2, 3, 0);

        // Ejecutar la predicción y verificar el resultado
        double prediction = knn.predict(data, target);
        
        // La predicción debería ser el promedio de los valores de los dos puntos más cercanos
        assertEquals(3.5, prediction, 0.001);
    }
    
    @Test
    public void testPredictWithSingleDataPoint() {
        // Configurar datos de entrenamiento con un solo punto
        List<KNNAlgorithm.DataPoint> data = Arrays.asList(
            new KNNAlgorithm.DataPoint(1, 1, 2)
        );

        // Crear instancia del algoritmo KNN con K=1
        KNNAlgorithm knn = new KNNAlgorithm(1);

        // Configurar el punto de destino
        KNNAlgorithm.DataPoint target = new KNNAlgorithm.DataPoint(1, 1, 0);

        // Ejecutar la predicción y verificar el resultado
        double prediction = knn.predict(data, target);
        
        // La predicción debería ser igual al valor del único punto de entrenamiento
        assertEquals(2.0, prediction, 0.001);
    }
}