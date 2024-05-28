import numpy as np
import pandas as pd

from src.Poblacion import Poblacion


class AG:

    # Contructor
    def __init__(self, datos_train, datos_test, seed, nInd, maxIter):
        self.datos_train = datos_train
        self.datos_test = datos_test
        self.seed = seed
        self.nInd = nInd
        self.maxIter = maxIter


    # Función para ejecutar el AG
    def run(self):
        np.random.seed(self.seed)

        datos = pd.read_csv(self.datos_train) 

        # Inicializar población
        poblacion = Poblacion(self.nInd, datos.shape[1]-1)   # nAtributos = nColumnas - 1
        poblacion_inicial = poblacion.initial()

        



        # SOLUCIÓN TEMPORAL
        ind = [0.5, -0.3, 0.1]  
        y_pred = [1.5, 2.3, -0.9]  
        return ind, y_pred

