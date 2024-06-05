import numpy as np
import pandas as pd
import math

from src.Poblacion import Poblacion
from src.Fitness import Fitness
from src.Padres import Padres
from src.Cruce import Cruce
from src.Mutacion import Mutacion
from src.Prediccion import Prediccion


class AG:

    # constructor
    def __init__(self, datos_train, datos_test, seed, num_ind, max_iter, verbose, population_method, crossover_method):
        self.datos_train = datos_train
        self.datos_test = datos_test
        self.seed = seed
        self.nInd = num_ind
        self.maxIter = max_iter
        self.verbose = verbose
        self.method = population_method
        self.crossover_method= crossover_method



    # función para ejecutar el AG
    def run(self):
        np.random.seed(self.seed)       # semilla para reproducibilidad

        # --------------------------------------------------------------------------------------
        # ---------------DATOS DE ENTRENAMIENTO Y PRIMERA ITERACIÓN DEL ALGORITMO---------------
        datos = pd.read_csv(self.datos_train, na_values=0)     
        nAtrib = datos.shape[1]-1      # nAtributos = nColumnas - 1

        # inicializar población
        poblacion = Poblacion(self.nInd, nAtrib, self.verbose,self.method, self.datos_train)
        poblacion_inicial = poblacion.initial()

        # fitness de la población inicial
        dicc_fitness = {}
        fitness_p_i = Fitness(datos, poblacion_inicial, self.nInd)
        fitness_poblacion_inicial = fitness_p_i.fitness_poblacion(dicc_fitness)
        
        # generamos los padres de la población inicial
        k = 3
        padres = Padres(fitness_poblacion_inicial, poblacion_inicial, self.nInd)
        seleccion_padres = padres.seleccion_padres_por_torneo(k)

        # generamos los hijos cruzando los padres
        
        probabilidad_no_cruce = 0.2
        marca=0
        cruce = Cruce(seleccion_padres, self.nInd, probabilidad_no_cruce,fitness_poblacion_inicial,self.crossover_method,marca,self.verbose)
        hijos_cruzados = cruce.cruzar()
        marca=1

        # generamos los hijos mutando los hijos cruzados
        mutacion = Mutacion(hijos_cruzados, fitness_poblacion_inicial)
        hijos_mutados = mutacion.mutar(np.min(fitness_poblacion_inicial))

        # poblacion a iterar
        poblacion_a_iterar = hijos_mutados
        tasa_elitismo = 0.2


        # --------------------------------------------------------------------------------------
        # -------------------------------BUCLE (Nº ITERACIONES)---------------------------------
        for i in range(0, self.maxIter):
            # fitness de la población a iterar
            fitness_h_m = Fitness(datos, poblacion_a_iterar, self.nInd)
            fitness_hijos_mutados = fitness_h_m.fitness_poblacion(dicc_fitness)

            # trabajamos para obtener la nueva generación
            # elegimos los mejores individuos (20%) de la población actual (elitismo)
            numero_individuos_elitismo = math.ceil(tasa_elitismo * self.nInd)
            indices_mejores_individuos = np.argsort(fitness_hijos_mutados)[:numero_individuos_elitismo]
            mejores_individuos = np.zeros((numero_individuos_elitismo, 2 * nAtrib + 1))
            for j in range(0, indices_mejores_individuos.size):                 # guardamos los mejores individuos para la nueva poblacion
                mejores_individuos[j] = poblacion_a_iterar[indices_mejores_individuos[j]]
                
            if(self.verbose):
               print(f"El mejor individuo de la poblacion {i} es: {poblacion_a_iterar[indices_mejores_individuos[0]]} con fitness: {fitness_hijos_mutados[np.argmin(fitness_hijos_mutados)]} ")
            
            # generamos los demas individuos de la nueva poblacion mediante torneo de la población anterior
            numIndivGenerar = self.nInd - numero_individuos_elitismo
            completamos_generacion = Padres(fitness_hijos_mutados, poblacion_a_iterar, numIndivGenerar)
            otra_parte_generacion_falta_eliminar = completamos_generacion.seleccion_padres_por_torneo(k)
            otra_parte_generacion = otra_parte_generacion_falta_eliminar[:numIndivGenerar]

            # unimos los dos grupos de individuos (los mejores y los seleccionados por torneo para rellenar la nueva población)
            seleccion_padres1 = np.concatenate((mejores_individuos, otra_parte_generacion))

            # generamos los hijos cruzando los padres
            probabilidad_no_cruce = 0.2
            cruce1 = Cruce(seleccion_padres1, self.nInd, probabilidad_no_cruce,fitness_hijos_mutados,self.crossover_method,marca,self.verbose)
            hijos_cruzados = cruce1.cruzar()
            
            # generamos los hijos mutando los hijos cruzados
            mutacion1 = Mutacion(hijos_cruzados, fitness_poblacion_inicial)
            hijos_mutados1 = mutacion1.mutar(np.min(fitness_hijos_mutados))

            # poblacion a iterar
            poblacion_a_iterar = hijos_mutados1


        # --------------------------------------------------------------------------------------
        # ------------------------------MEJOR SOLUCIÓN ENCONTRADA-------------------------------
        fitness_p_f = Fitness(datos, poblacion_a_iterar, self.nInd)
        fitness_poblacion_final = fitness_p_f.fitness_poblacion(dicc_fitness)
        mejor_individuo_encontrado = poblacion_a_iterar[np.argmin(fitness_poblacion_final)]


        # --------------------------------------------------------------------------------------
        # --------------------------SOLUCIÓN SOBRE EL CONJUNTO DE TEST--------------------------
        datos_prediccion = pd.read_csv(self.datos_test, na_values=0)
        prediccion = Prediccion(datos_prediccion, mejor_individuo_encontrado)
        y_pred = prediccion.predecir()  


        # --------------------------------------------------------------------------------------
        # devolvemos la mejor solución encontrada y las predicciones sobre el conjunto de test
        return mejor_individuo_encontrado, y_pred
    

