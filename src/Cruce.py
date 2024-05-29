import numpy as np
import random
import math

class Cruce:

    def __init__(self, padres,numero_individuos):
        self.padres = padres
        self.numero_individuos= numero_individuos
           
    
    def cruce(self):
        hijos = np.empty_like(self.padres)
        
        #Primero seleccionamos los individuos que van a pasar sin cruces, es decir, pasan tal cual:
        #Seleccionamos el 20% aleatoriamente
        numero= math.ceil(self.numero_individuos * 0.2)
        indices_aleatorios = random.sample(range(0, self.padres.shape[0]), numero)

        #Los añadimos en las n primeras filas de hijos
        seleccionados = self.padres[indices_aleatorios]
        hijos[:numero] = seleccionados
        
        #Para el resto de individuos cruzamos aleatoriamente dos de los padres
        while numero < self.numero_individuos-1:

            #Seleccionamos dos padres aletoriamente
            indices = random.sample(range(0, self.padres.shape[0]), 2)
            selec_cruce = self.padres[indices]

            #Seleccionamos por qué parte de la lista vamos a partir aleatoriamente
            num = random.randint(0, self.padres.shape[1]-1)

            #Seleccionamos los n primeros elementos del 1
            padre1 = selec_cruce[0][:num]

            # y los siguientes del segundo
            padre2 = selec_cruce[1][num:]

            # Crear un nuevo array combinando los elementos seleccionados
            hijo = np.concatenate((padre1, padre2))

            numero+=1
            hijos[numero] = hijo

        return hijos



    
