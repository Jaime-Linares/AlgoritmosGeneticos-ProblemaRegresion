import numpy as np
import random
import math

class Mutacion:

    def __init__(self, hijos, fitness, probabilidad_baja=0.1, probabilidad_alta=0.3, umbral_estancamiento=5):
        self.hijos = hijos
        self.probabilidad_baja = probabilidad_baja
        self.probabilidad_alta = probabilidad_alta
        self.probabilidad_actual = probabilidad_baja
        self.umbral_estancamiento = umbral_estancamiento
        self.generaciones_sin_mejora = 0
        self.mejor_fitness_anterior = None
        self.fitness = fitness


    def mutar(self, mejor_fitness_actual):
        # comprobamos si la anterior generación el fitness mejoró o es la 1 iteración
        if self.mejor_fitness_anterior is None or mejor_fitness_actual < self.mejor_fitness_anterior:
            # si es así, establecemos la probabilidad de mutación en baja, y reseteamos el contador de poblaciones sin mejora
            self.generaciones_sin_mejora = 0
            self.probabilidad_actual = self.probabilidad_baja
            self.mejor_fitness_anterior = mejor_fitness_actual
        else:
            # si no es así, aumentamos el contador de poblaciones sin mejora, y cuando este supere un umbral,
            # se cambiará la probabilidad de mutación a una mayor para evitar estancamientos
            self.generaciones_sin_mejora += 1
            if self.generaciones_sin_mejora >= self.umbral_estancamiento:
                self.probabilidad_actual = self.probabilidad_alta

        # elegimos el número de individuos que mutarán y cuáles van a ser
        hijos_mutados = np.copy(self.hijos)
        numero_mutaciones = math.ceil(self.probabilidad_actual * self.hijos.shape[0])
        
        # aseguramos que el mejor individuo no se pueda mutar
        mejor_individuo= 0
        indices_aleatorios = [i for i in range(self.hijos.shape[0]) if i != mejor_individuo]
        indices_seleccionados = random.sample(indices_aleatorios, numero_mutaciones)

        # mutamos los individuos seleccionados
        for i in indices_seleccionados:
            # para cada individuo se mutará un número de genes aleatorios que van desde 1 hasta el 10% del total de genes de individuo
            num_genes_mutar = random.randint(1, math.ceil(self.hijos.shape[1] * 0.1))
            for _ in range(num_genes_mutar):
                gen = random.randint(0, self.hijos.shape[1] - 1)  # Elegimos el gen que mutará
                
                if gen in range(0, 2 * self.hijos.shape[1], 2) or gen == 2 * self.hijos.shape[1]:
                    # es una constante o el término independiente, cambiamos el valor por otro valor random entre -50 y 50
                    hijos_mutados[i, gen] = (hijos_mutados[i, gen] + np.random.uniform(-50, 50)) / 2
                
                elif gen in range(1, 2 * self.hijos.shape[1], 2):
                    # es posición impar (exponente) excepto el término independiente cambiamos dependiendo de ...
                    if self.hijos[i, gen] > -1 and self.hijos[i, gen] < 5:  # si el exponente está entre -1 y 5, aleatoriamente se suma o resta 1
                        numero_aleatorio = random.randint(0, 1)
                        if numero_aleatorio == 0:
                            hijos_mutados[i, gen] += 1
                        else:
                            hijos_mutados[i, gen] -= 1
                    elif self.hijos[i, gen] == -1:                          # si el exponente es -1, se le suma 1
                        hijos_mutados[i, gen] += 1
                    elif self.hijos[i, gen] == 5:                           # si el exponente es 5, se le resta 1
                        hijos_mutados[i, gen] -= 1
                    else:
                        hijos_mutados[i, gen] = 2

        return hijos_mutados
