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
        
        #Para cada individuo se mutará un número de genes aleatorios que van desde 1 hasta el 10% del total de genes de individuo
        
        

        # mutamos los individuos seleccionados
        for i in indices_aleatorios:
            num_genes_mutar= random.randint(1, math.ceil(self.hijos.shape[1]*0.1))
            j=0
            while j <= num_genes_mutar:
                j+=1
                gen = random.randint(0, self.hijos.shape[1]-1)      # elegimos el gen que mutará

                # si es una constante o el término independiente, cambiamos el valor por otro valor random entre -100 y 100
                if gen in range(0, 2 * self.hijos.shape[1], 2) or gen == 2 * self.hijos.shape[1]:  
                    hijos_mutados[i, gen] = (hijos_mutados[i,gen] + np.random.uniform(-50, 50))/2

                # si es posición impar (exponente) excepto el término independiente
                elif gen in range(1, 2 * self.hijos.shape[1], 2):    
                    if self.hijos[i, gen] >-1 and self.hijos[i, gen] <5:    # si el exponente está entre -1 y 5, aleatoriamente se suma o resta 1
                        numero_aleatorio = random.randint(0, 1)
                        if(numero_aleatorio==0):
                            hijos_mutados[i, gen] += 1
                        else:
                            hijos_mutados[i, gen] -= 1
                    elif self.hijos[i, gen] == -1:                          # si el exponente es -1, se le suma 1
                        hijos_mutados[i, gen] += 1
                    else:                                                   # si el exponentes 5, se le resta 1
                        hijos_mutados[i, gen] -= 1 

        return hijos_mutados
    

    