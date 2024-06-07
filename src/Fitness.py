import numpy as np
import pandas as pd

class Fitness:
    def __init__(self, datos, poblacion, num_ind):
        self.datos = datos
        self.poblacion = poblacion
        self.num_ind = num_ind

    def fitness_poblacion(self, dicc_fitness):
        # Convertir la población a un array de numpy para operaciones vectorizadas
        poblacion_array = np.array(self.poblacion)
        
        # Evaluar la población completa
        fitness_values = self.evaluate_population(poblacion_array)
        
        # Actualizar el diccionario con los valores de fitness
        for ind, fitness_value in zip(poblacion_array, fitness_values):
            dicc_fitness[tuple(ind)] = fitness_value
        
        return fitness_values

    def evaluate_population(self, population):
        # Obtener X y y de los datos
        X = self.datos.iloc[:, :-1].values
        y = self.datos.iloc[:, -1].values
        
        # Descomponer la población en coeficientes, exponentes y la constante
        num_atrib = population.shape[1]
        coeficientes = population[:, :num_atrib][:, ::2][:, :-1]
        exponentes = population[:, :num_atrib][:, 1::2]
        constantes = population[:, -1]
        
        
        
        # Ajustar las dimensiones para la operación de broadcast
        X_expanded = X[:, np.newaxis, :]  # Shape (num_samples, 1, num_atrib)
        coeficientes_expanded = coeficientes[np.newaxis, :, :]  # Shape (1, num_ind, num_atrib)
        exponentes_expanded = exponentes[np.newaxis, :, :]  # Shape (1, num_ind, num_atrib)
        
        # Calcular predicciones vectorizadas
        predictions = np.sum(coeficientes_expanded * np.power(X_expanded, exponentes_expanded), axis=2) + constantes
        
        # Calcular el MSE para cada individuo
        mse = np.mean((predictions - y[:, np.newaxis]) ** 2, axis=0)
        
        return mse
