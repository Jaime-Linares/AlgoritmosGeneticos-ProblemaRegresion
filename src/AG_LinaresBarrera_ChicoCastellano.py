import numpy as np
import pandas as pd
import math

from src.Poblacion import Poblacion
from src.Padres import Padres
from src.Cruce import Cruce
from src.Mutacion import Mutacion
from src.Prediccion import Prediccion


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
        nAtrib = datos.shape[1]-1      # nAtributos = nColumnas - 1

        # inicializar población
        poblacion = Poblacion(self.nInd, nAtrib)   
        poblacion_inicial = poblacion.initial()

        # fitness de la población inicial
        fitness_poblacion_inicial = self.__fitness_poblacion(datos, poblacion_inicial)
        
        # generamos los padres de la población inicial
        k = 3
        padres = Padres(fitness_poblacion_inicial, poblacion_inicial, self.nInd)
        seleccion_padres = padres.seleccion_padres_por_torneo(k)

        # generamos los hijos cruzando los padres
        probabilidad_no_cruce = 0.2
        cruce = Cruce(seleccion_padres, self.nInd, probabilidad_no_cruce, None)
        hijos_cruzados = cruce.cruzar()

        # generamos los hijos mutando los hijos cruzados
        probabilidad_mutacion = 0.1
        hijos = Mutacion(hijos_cruzados, probabilidad_mutacion)
        hijos_mutados = hijos.mutar()

        # poblacion a iterar
        poblacion_a_iterar = hijos_mutados
        tasa_elitismo = 0.2

        # --------------------------------------------------------------------------------------
        # -------------------------------BUCLE (Nº ITERACIONES)---------------------------------
        for i in range(0, self.maxIter):
            # fitness de la población a iterar
            fitness_hijos_mutados = self.__fitness_poblacion(datos, poblacion_a_iterar)

            # trabajamos para obtener la nueva generación
            # elegimos los mejores individuos (20%) de la población actual (elitismo)
            numero_individuos_elitismo = math.ceil(tasa_elitismo * self.nInd)
            indices_mejores_individuos = np.argsort(fitness_hijos_mutados)[:numero_individuos_elitismo]
            mejores_individuos = np.zeros((numero_individuos_elitismo, 2 * nAtrib + 1))
            for i in range(0, indices_mejores_individuos.size):                    # guardamos los mejores individuos para la nueva poblacion
                mejores_individuos[i] = poblacion_a_iterar[indices_mejores_individuos[i]]
            

            # generamos los demas individuos de la nueva poblacion mediante cruces y mutaciones de la generación anterior
            # generamos individuos mediante cruces
            cruce = Cruce(mejores_individuos, self.nInd , probabilidad_no_cruce, poblacion_inicial)
            poblacion_cruzada = cruce.cruzar()

            # generamos individuos mediante mutaciones
            mutacion = Mutacion(poblacion_cruzada, probabilidad_mutacion)
            poblacion_mutada = mutacion.mutar()

            # obtenemos la nueva generación (nueva poblacion)
            fitness_nueva_poblacion = self.__fitness_poblacion(datos, poblacion_mutada)

            # generamos los padres de la población inicial
            padres = Padres(fitness_nueva_poblacion, poblacion_mutada, self.nInd)
            seleccion_padres = padres.seleccion_padres_por_torneo(k)

            # generamos los hijos cruzando los padres
            probabilidad_no_cruce = 0.2
            cruce1 = Cruce(seleccion_padres, self.nInd, probabilidad_no_cruce, None)
            hijos_cruzados = cruce1.cruzar()
            
            # generamos los hijos mutando los hijos cruzados
            probabilidad_mutacion = 0.1
            hijos = Mutacion(hijos_cruzados, probabilidad_mutacion)
            hijos_mutados = hijos.mutar()

            # poblacion a iterar
            poblacion_a_iterar = hijos_mutados


        # --------------------------------------------------------------------------------------
        # ------------------------------MEJOR SOLUCIÓN ENCONTRADA-------------------------------
        fitness_poblacion_final = self.__fitness_poblacion(datos, poblacion_a_iterar)
        mejor_individuo_encontrado = poblacion_a_iterar[np.argmin(fitness_poblacion_final)]


        # --------------------------------------------------------------------------------------
        # --------------------------SOLUCIÓN SOBRE EL CONJUNTO DE TEST--------------------------
        datos_prediccion = pd.read_csv(self.datos_test, na_values=0)
        prediccion = Prediccion(datos_prediccion ,mejor_individuo_encontrado)
        y_pred = prediccion.predecir()  


        # --------------------------------------------------------------------------------------
        # devolvemos la mejor solución encontrada y las predicciones sobre el conjunto de test
        return mejor_individuo_encontrado, y_pred
    


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


