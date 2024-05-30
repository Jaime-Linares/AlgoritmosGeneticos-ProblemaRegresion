import numpy as np
import random
import math


class Cruce:

    def __init__(self, padres, numero_individuos, probabilidad_no_cruce, poblacion):
        self.padres = padres
        self.numero_individuos= numero_individuos
        self.probabilidad_no_cruce = probabilidad_no_cruce
        self.poblacion= poblacion
           
    
    def cruzar(self):
        if (self.poblacion is None):
            hijos = np.empty_like(self.padres)
        else:
            hijos = np.empty_like(self.poblacion)
            
        # primero seleccionamos los individuos que van a pasar sin cruces, es decir, pasan tal cual:
        # seleccionamos el 20% aleatoriamente
        numero= math.ceil(self.numero_individuos * self.probabilidad_no_cruce)
        indices_aleatorios = random.sample(range(0, self.padres.shape[0]), numero)

        # los añadimos en las n primeras filas de hijos
        seleccionados = self.padres[indices_aleatorios]
        hijos[:numero] = seleccionados
            
        # para el resto de individuos cruzamos aleatoriamente dos de los padres
        while numero < self.numero_individuos:

            # seleccionamos dos padres aletoriamente
            indices = random.sample(range(0, self.padres.shape[0]), 2)
            selec_cruce = self.padres[indices]

            # seleccionamos por qué parte de la lista vamos a partir aleatoriamente
            num = random.randint(0, self.padres.shape[1]-1)

            # seleccionamos los n primeros elementos del 1
            padre1 = selec_cruce[0][:num]

            # y los siguientes del segundo
            padre2 = selec_cruce[1][num:]

            # crear un nuevo array combinando los elementos seleccionados
            hijo = np.concatenate((padre1, padre2))

            hijos[numero] = hijo
            numero+=1
            
        return hijos


