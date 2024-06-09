import numpy as np
import random
import math


class Mutacion:

    def __init__(self, hijos, probabilidad_baja, probabilidad_alta, umbral_estancamiento):
        self.hijos = hijos
        self.probabilidad_baja = probabilidad_baja
        self.probabilidad_alta = probabilidad_alta
        self.probabilidad_actual = probabilidad_baja
        self.umbral_estancamiento = umbral_estancamiento
        self.mejor_fitness_anterior = None


    def mutar(self, mejor_fitness_actual, generaciones_sin_mejora):    
        hijos_mutados = np.copy(self.hijos)

        # comprobamos si la anterior generación el fitness mejoró o si estamos en la primera iteración
        if self.mejor_fitness_anterior is None or mejor_fitness_actual < self.mejor_fitness_anterior:
            # si es así, establecemos la probabilidad de mutación en baja, y reseteamos el contador de poblaciones sin mejora
            generaciones_sin_mejora = 0
            self.probabilidad_actual = self.probabilidad_baja
            self.mejor_fitness_anterior = mejor_fitness_actual
        else:
            # si no es así, aumentamos el contador de poblaciones sin mejora, y cuando este supere un umbral,
            # se cambiará la probabilidad de mutación a una mayor para evitar estancamientos
            generaciones_sin_mejora += 1
            if generaciones_sin_mejora >= self.umbral_estancamiento:
                self.probabilidad_actual = self.probabilidad_alta

        # aseguramos que el mejor individuo no se pueda mutar
        indice_mejor_individuo = 0
        indices_aleatorios = [i for i in range(self.hijos.shape[0]) if i != indice_mejor_individuo]

        # elegimos el número de individuos que mutarán y que individuos serán
        numero_mutaciones = math.ceil(self.probabilidad_actual * self.hijos.shape[0])
        indices_seleccionados = random.sample(indices_aleatorios, numero_mutaciones)

        # mutamos los individuos seleccionados
        for i in indices_seleccionados:
            # para cada individuo se mutará un número de genes aleatorios que van desde 1 hasta el 10% del total de genes de individuo
            num_genes_mutar = random.randint(1, math.ceil(self.hijos.shape[1] * 0.1))

            # mutamos el/los gen/genes
            for n in range(0, num_genes_mutar):
                # elegimos el gen que mutará
                gen = random.randint(0, self.hijos.shape[1] - 1)
                
                # si el gen corresponde a una constante o al término independiente, cambiamos el valor por otro valor random entre -50 y 50
                if gen in range(0, 2 * self.hijos.shape[1], 2) or gen == 2 * self.hijos.shape[1]:
                    hijos_mutados[i, gen] = (hijos_mutados[i, gen] + np.random.uniform(-50, 50)) / 2
                
                # si el gen corresponde a un exponente (posición impar excepto el término independiente) ...
                elif gen in range(1, 2 * self.hijos.shape[1], 2):
                    # si el exponente está entre -1 y 5, aleatoriamente se suma o resta 1
                    if self.hijos[i, gen] > -1 and self.hijos[i, gen] < 5:
                        numero_aleatorio = random.randint(0, 1)
                        if numero_aleatorio == 0:
                            hijos_mutados[i, gen] += 1
                        else:
                            hijos_mutados[i, gen] -= 1
                    # si el exponente es -1, se le suma 1
                    elif self.hijos[i, gen] == -1:
                        hijos_mutados[i, gen] += 1
                    # si el exponente es 5, se le resta 1
                    else:                                               
                        hijos_mutados[i, gen] -= 1
                    
        return hijos_mutados
    
    
