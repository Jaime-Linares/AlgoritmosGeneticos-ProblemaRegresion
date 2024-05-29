import numpy as np
import pandas as pd

from src.Poblacion import Poblacion
from src.Padres import Padres
from src.Cruce import Cruce
from src.Mutacion import Mutacion


class AG:

    # contructor
    def __init__(self, datos_train, datos_test, seed, nInd, maxIter):
        self.datos_train = datos_train
        self.datos_test = datos_test
        self.seed = seed
        self.nInd = nInd
        self.maxIter = maxIter



    # función para ejecutar el AG
    def run(self):
        np.random.seed(self.seed)       # semilla para reproducibilidad

        # --------------------------------------------------------------------------------------
        # ---------------DATOS DE ENTRENAMIENTO Y PRIMERA ITERACIÓN DEL ALGORITMO---------------
        datos = pd.read_csv(self.datos_train, na_values=0)     

        # inicializar población
        poblacion = Poblacion(self.nInd, datos.shape[1]-1)   # nAtributos = nColumnas - 1
        poblacion_inicial = poblacion.initial()

        # fitness de la población inicial
        fitness_poblacion_inicial = self.__fitness_poblacion(datos, poblacion_inicial)
        
        # generamos los padres de la población inicial
        k = 3
        padres = Padres(fitness_poblacion_inicial, poblacion_inicial, self.nInd, k)
        seleccion_padres = padres.seleccion_padres()

        # generamos los hijos cruzando los padres
        probabilidad_no_cruce = 0.2
        cruce = Cruce(seleccion_padres, self.nInd, probabilidad_no_cruce)
        hijos_cruzados = cruce.cruzar()

        # generamos los hijos mutando los hijos cruzados
        probabilidad_mutacion = 0.1
        hijos = Mutacion(hijos_cruzados, probabilidad_mutacion)
        hijos_mutados = hijos.mutar()

        # seleccionamos siguiente población
        

        # --------------------------------------------------------------------------------------
        # -------------------------------BUCLE (Nº ITERACIONES)---------------------------------
        #for i in range(0, self.maxIter):
        

        # --------------------------------------------------------------------------------------
        # ------------------------------MEJOR SOLUCIÓN ENCONTRADA-------------------------------
        ind = [0.5, -0.3, 0.1] 


        # --------------------------------------------------------------------------------------
        # --------------------------SOLUCIÓN SOBRE EL CONJUNTO DE TEST--------------------------
        y_pred = [1.5, 2.3, -0.9]  


        # --------------------------------------------------------------------------------------
        # devolvemos la mejor solución encontrada y las predicciones sobre el conjunto de test
        return ind, y_pred
    


    # función para calcular el fitness de la población
    def __fitness_poblacion(self, datos, poblacion):
        matrix_errors = np.zeros((datos.shape[0], poblacion.shape[0]))   # matriz de ceros de dimension nDatos x nInd

        # obtenemos para cada ecuacion una lista con el error (diferencia al cuadrado) de cada uno de
        # los individuos de la población
        for i in range(0, datos.shape[0]):
            for j in range(0, poblacion.shape[0]):
                matrix_errors[i, j] = self.__evalua_individuo(datos.iloc[i], poblacion[j, :])
        
        # calculamos el fitness de cada individuo como la suma de sus errores en cada una de las ecuaciones
        res = np.zeros(self.nInd)
        for i in range(0, self.nInd):
            res[i] = np.sum(matrix_errors[:, i])   # sumamos los errores de cada ecuacion para el individuo i

        return res


    # función para calcular la diferenca entre el valor real y el valor obtenido para tal individuo
    # en tal ecuacion
    def __evalua_individuo(self, valores, parametros):
        res = 0
        calculos_prohibidos = False     # para evitar indeterminaciones como 0 elevado a 0

        y_real = valores.iloc[-1]
        y_aprox = 0
        
        # calculamos la y_aprox (la ecuacion) para el individuo
        for i in range(0, (parametros.size)-1, 2):    # primero calculamos la suma de los terminos 
            x_i = i // 2        # división entera
            par_i = i
            
            if valores.iloc[x_i] == 0 and parametros[par_i+1] == 0:     # 0 elevado a 0
                y_aprox += 0
                calculos_prohibidos = True
                break
            elif valores.iloc[x_i] == 0 and parametros[par_i+1] < 0:    # division por 0
                y_aprox += 0
                calculos_prohibidos = True
                break
            else:                                                       # calculo normal de la ecuación
                potencia = valores.iloc[x_i] ** parametros[par_i+1]
                y_aprox += parametros[par_i] * potencia

        y_aprox += parametros[-1]   # sumamos el término independiente

        if calculos_prohibidos:
            res = 1000000000000
        else:
            res = (y_real - y_aprox) ** 2   # calculamos el cuadrado de la diferencia

        return res

