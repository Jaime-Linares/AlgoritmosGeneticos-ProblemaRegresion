import numpy as np
import pandas as pd
import math

from src.Poblacion import Poblacion
from src.Fitness import Fitness
from src.Padres import Padres
from src.Cruce import Cruce
from src.Mutacion import Mutacion
from src.Prediccion import Prediccion


class AG1:

    # constructor
    def __init__(self, datos_train, datos_test, seed, num_ind, max_iter, verbose, population_method, crossover_method,probabilidad_baja, probabilidad_alta, k, tasa_elitismo, tasa_no_cruce):
        self.datos_train = datos_train
        self.datos_test = datos_test
        self.seed = seed
        self.num_ind = num_ind
        self.max_iter = max_iter
        self.verbose = verbose
        self.population_method = population_method
        self.crossover_method= crossover_method
        self.probabilidad_baja = probabilidad_baja
        self.probabilidad_alta = probabilidad_alta
        self.k = k
        self.tasa_elitismo = tasa_elitismo
        self.tasa_no_cruce = tasa_no_cruce



    # función para ejecutar el AG
    def run(self):
        np.random.seed(self.seed)       # semilla para reproducibilidad

        # --------------------------------------------------------------------------------------
        # ---------------DATOS DE ENTRENAMIENTO Y PRIMERA ITERACIÓN DEL ALGORITMO---------------
        datos = pd.read_csv(self.datos_train, na_values=0)     
        num_atrib = datos.shape[1]-1      # numAtributos = numColumnas - 1

        # inicializar población
        poblacion = Poblacion(self.num_ind, num_atrib, self.verbose, self.population_method, self.datos_train)
        poblacion_inicial = poblacion.initial()

        # fitness de la población inicial
        dicc_fitness = {}
        fitness_p_i = Fitness(datos, poblacion_inicial, self.num_ind)
        fitness_poblacion_inicial = fitness_p_i.fitness_poblacion(dicc_fitness)
        
        # generamos los padres de la población inicial
        k = self.k
        padres = Padres(fitness_poblacion_inicial, poblacion_inicial, self.num_ind)
        seleccion_padres = padres.seleccion_padres_por_torneo(k)

        # generamos los hijos cruzando los padres
        probabilidad_no_cruce = self.tasa_no_cruce
        mark = 0
        cruce = Cruce(seleccion_padres, self.num_ind, probabilidad_no_cruce, fitness_poblacion_inicial, self.crossover_method, mark, self.verbose)
        hijos_cruzados = cruce.cruzar()
        mark = 1
        
        # generamos los hijos mutando los hijos cruzados
        generaciones_sin_mejora=0
        probabilidad_baja=self.probabilidad_baja
        probabilidad_alta=self.probabilidad_alta
        umbral_estancamiento=5
        mutacion = Mutacion(hijos_cruzados, fitness_poblacion_inicial,probabilidad_baja, probabilidad_alta, umbral_estancamiento)
        hijos_mutados = mutacion.mutar(np.min(fitness_poblacion_inicial), generaciones_sin_mejora)

        # poblacion a iterar y la tasa de elitismo usada para generar las nuevas generaciones
        poblacion_a_iterar = hijos_mutados
        tasa_elitismo = self.tasa_elitismo


        # --------------------------------------------------------------------------------------
        # -------------------------------BUCLE (Nº ITERACIONES)---------------------------------
        for i in range(0, self.max_iter):
            # fitness de la población a iterar
            fitness_h_m = Fitness(datos, poblacion_a_iterar, self.num_ind)
            fitness_hijos_mutados = fitness_h_m.fitness_poblacion(dicc_fitness)

            # trabajamos para obtener la nueva generación
            # elegimos los mejores individuos (20%) de la población actual (elitismo)
            numero_individuos_elitismo = math.ceil(tasa_elitismo * self.num_ind)
            indices_mejores_individuos = np.argsort(fitness_hijos_mutados)[:numero_individuos_elitismo]
            mejores_individuos = np.zeros((numero_individuos_elitismo, 2 * num_atrib + 1))
            for j in range(0, indices_mejores_individuos.size):                 # guardamos los mejores individuos para la nueva poblacion
                mejores_individuos[j] = poblacion_a_iterar[indices_mejores_individuos[j]]
                
            if(self.verbose):
               print(f"El mejor individuo de la poblacion {i} es: {poblacion_a_iterar[indices_mejores_individuos[0]]}, con fitness: {fitness_hijos_mutados[np.argmin(fitness_hijos_mutados)]}")
            
            # generamos los demas individuos de la nueva poblacion mediante torneo de la población anterior
            num_indiv_generar_para_completar = self.num_ind - numero_individuos_elitismo
            completamos_generacion = Padres(fitness_hijos_mutados, poblacion_a_iterar, num_indiv_generar_para_completar)
            otra_parte_generacion_falta_eliminar = completamos_generacion.seleccion_padres_por_torneo(k)
            otra_parte_generacion = otra_parte_generacion_falta_eliminar[:num_indiv_generar_para_completar]

            # unimos los dos grupos de individuos (los mejores y los seleccionados por torneo para rellenar la nueva población)
            seleccion_padres1 = np.concatenate((mejores_individuos, otra_parte_generacion))

            # generamos los hijos cruzando los padres
            probabilidad_no_cruce = self.tasa_no_cruce
            cruce1 = Cruce(seleccion_padres1, self.num_ind, probabilidad_no_cruce, fitness_hijos_mutados, self.crossover_method, mark, self.verbose)
            hijos_cruzados1 = cruce1.cruzar()
            
            # generamos los hijos mutando los hijos cruzados
            mutacion1 = Mutacion(hijos_cruzados1, fitness_hijos_mutados,probabilidad_baja, probabilidad_alta, umbral_estancamiento)   
            hijos_mutados1 = mutacion1.mutar(np.min(fitness_hijos_mutados), generaciones_sin_mejora)

            # poblacion a iterar
            poblacion_a_iterar = hijos_mutados1


        # --------------------------------------------------------------------------------------
        # ------------------------------MEJOR SOLUCIÓN ENCONTRADA-------------------------------
        fitness_p_f = Fitness(datos, poblacion_a_iterar, self.num_ind)
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
    

