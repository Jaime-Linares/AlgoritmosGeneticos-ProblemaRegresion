import numpy as np
import random


class Padres:

    def __init__(self, fitness, individuos, numIndividuos):
            self.fitness = fitness
            self.individuos = individuos
            self.numIndividuos = numIndividuos
            
    
    def seleccion_padres_por_torneo(self, k):
        padres = np.empty_like(self.individuos)

        for i in range(0, self.numIndividuos):
            # escogemos tres individuos aleatorios
            indices_aleatorios = random.sample(range(0, self.individuos.shape[0]), k)
            # obtenemos sus fitness
            fitness_seleccionados = [self.fitness[pos] for pos in indices_aleatorios]
            # escogemos el individuo con mejor fitness de los k elegidos (que será el menor)
            pos_mayor_fitness = 0
            indiviudo_mayor_fitness = self.individuos[indices_aleatorios[pos_mayor_fitness], :]
            for j in range(1, k):
                if fitness_seleccionados[j] < fitness_seleccionados[pos_mayor_fitness]:
                    pos_mayor_fitness = j
                    indiviudo_mayor_fitness = self.individuos[indices_aleatorios[pos_mayor_fitness], :]
            # guardamos el individuo con mayor fitness como uno de los padres
            padres[i] = indiviudo_mayor_fitness
        
        return padres
            

            