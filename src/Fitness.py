import numpy as np


class Fitness:

    def __init__(self, datos, poblacion):
        self.datos = datos
        self.poblacion = poblacion


    # función para calcular el fitness la población
    def fitness_poblacion(self):
        # convertir la población a un array de numpy para operaciones vectorizadas
        poblacion_array = np.array(self.poblacion)
        # evaluar la población completa
        fitness_values = self.evaluate_population(poblacion_array)
        # donde no pueda calcularse el fitness (porque es un valor muy grande), asignar un valor muy alto (infinito)
        fitness_values = np.where(np.isnan(fitness_values), float('inf'), fitness_values)

        return fitness_values


    # función para evaluar la población
    def evaluate_population(self, population):
        # obtener X e y de los datos
        X = self.datos.iloc[:, :-1].values
        y = self.datos.iloc[:, -1].values
        
        # descomponer la población en coeficientes, exponentes y la constante
        num_atrib = population.shape[1]
        coeficientes = population[:, :num_atrib][:, ::2][:, :-1]
        exponentes = population[:, :num_atrib][:, 1::2]
        constantes = population[:, -1]
        
        # ajustar las dimensiones para la operación de broadcast
        X_expanded = X[:, np.newaxis, :]                            # shape = (num_samples, 1, num_atrib)
        coeficientes_expanded = coeficientes[np.newaxis, :, :]      # shape = (1, num_ind, num_atrib)
        exponentes_expanded = exponentes[np.newaxis, :, :]          # shape = (1, num_ind, num_atrib)
        
        # calcular predicciones vectorizadas
        predictions = np.sum(coeficientes_expanded * np.power(X_expanded, exponentes_expanded), axis=2) + constantes
        
        # calcular el error para cada individuo, es decir, una media de la diferencia al cuadrado entre las predicciones
        # y los resultados reales para cada individuo de la población
        fitness = np.mean((predictions - y[:, np.newaxis]) ** 2, axis=0)
        
        return fitness
    

