import numpy as np
import math

class Padres:

    def __init__(self, fitness, individuos):
            self.fitness = fitness
            self.individuos = individuos
            
    
    def seleccion_padres(self,individuos,fitness):
        #Generamos un diccionario donde las claves son los individuos y el valor el fitness de cada uno
        pares = zip(individuos, fitness )
        diccionario = dict(pares)

        #Ordenamos en función del valor de su fitness (de menor a mayor)
        diccionario_ordenado = dict(sorted(diccionario.items(), key=lambda item: item[0]))

        #seleccionamo el 20% de los individuos con menos fitness
        numero_individuos= math.ceil(len(self.individuos) * 0.4)

        #Seleccionamos las claves ordenadas
        claves_ordenadas = sorted(diccionario.keys())

        #Obtener los valores asociados a las primeras 10 claves
        padres = [diccionario[clave] for clave in claves_ordenadas[:numero_individuos]]
        return padres
