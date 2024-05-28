import numpy as np


class Poblacion:

    def __init__(self, nInd, nAtrib):
        self.nInd = nInd
        self.nAtrib = nAtrib


    # Función para crear la población inicial
    # matriz con valoresde numero decimales aleatorios en el rango [-1000, 1000] de dimension nInd x (2*nAtrib)+1
    def initial(self):
        return np.random.uniform(-1000, 1001, size=(self.nInd, 2*(self.nAtrib)+1))
