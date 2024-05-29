import numpy as np
import random
import math


class Mutacion:

    def __init__(self, hijos, probabilidad):
        self.hijos = hijos
        self.probabilidad = probabilidad


    def mutar(self):
        hijos_mutados = np.copy(self.hijos)

        # elegimos el numero de individuos que mutarán y cuales van a ser
        numero_mutaciones = math.ceil(self.probabilidad * self.hijos.shape[0])
        indices_aleatorios = random.sample(range(0, self.hijos.shape[0]), numero_mutaciones)
        # mutamos los individuos seleccionados
        for i in indices_aleatorios:
            gen = random.randint(0, self.hijos.shape[1]-1)      # elegimos el gen que mutará
            if gen in range(0, 2 * self.hijos.shape[1], 2) or gen == 2 * self.hijos.shape[1]:     # si es una constante o el término independiente
                if self.hijos[i, gen] >= 50:                            # si el gen es mayor o igual a 50 le restamos 50
                    hijos_mutados[i, gen] = self.hijos[i, gen] - 50
                elif self.hijos[i, gen] <= -50:                         # si el gen es menor o igual a -50 le sumamos 50
                    hijos_mutados[i, gen] = self.hijos[i, gen] + 50
                else:                                                   # si el gen está entre -50 y 50 lo dividimos por 2
                    hijos_mutados[i, gen] = self.hijos[i, gen] / 2
            elif gen in range(1, 2 * self.hijos.shape[1], 2):   # si es posición impar (exponente) excepto el término independiente
                if self.hijos[i, gen] == 0:             # si el gen es 0 el gen será 1    
                    hijos_mutados[i, gen] = 1
                else:                                   # si el gen es 1 el gen será 0
                    hijos_mutados[i, gen] = 0
        
        return hijos_mutados