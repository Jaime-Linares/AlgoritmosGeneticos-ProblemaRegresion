import numpy as np


class Poblacion:

    def __init__(self, nInd, nAtrib):
        self.nInd = nInd
        self.nAtrib = nAtrib


    # función para crear la población inicial
    # matriz con valores de numero decimales aleatorios de dimension nInd x (2*nAtrib)+1
    def initial(self):
        poblacion_inicial = np.zeros((self.nInd, 2 * self.nAtrib + 1))

        # números decimales aleatorios para las posiciones pares entre -100 y 100
        indices_pares = range(0, 2 * self.nAtrib, 2)
        poblacion_inicial[:, indices_pares] = np.random.uniform(-100, 100, size=(self.nInd, self.nAtrib))
        # números enteros aleatorios para las posiciones impares (excepto la última) entre -1 y 5
        indices_impares = range(1, 2 * self.nAtrib, 2)
        poblacion_inicial[:, indices_impares] = np.random.randint(-1, 5, size=(self.nInd, self.nAtrib))
        # números decimales aleatorio en el rango para la última posición impar entre -100 y 100
        poblacion_inicial[:, 2*self.nAtrib] = np.random.uniform(-100, 100, size=self.nInd)

        return poblacion_inicial


