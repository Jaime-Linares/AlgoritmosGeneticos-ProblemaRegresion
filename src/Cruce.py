import numpy as np
import random
import math


class Cruce:

    def __init__(self, padres, num_ind, probabilidad_no_cruce, fitness, crossover_method, mark, verbose):
        self.padres = padres
        self.num_ind = num_ind
        self.probabilidad_no_cruce = probabilidad_no_cruce
        self.fitness = fitness
        self.metodo = crossover_method
        self.mark = mark
        self.verbose= verbose


    # método para elegir el tipo de cruce que vamos a usar
    def cruzar(self):
        if self.metodo == "dos_puntos":
            if(self.verbose and self.mark==0):
                print("Metodo de cruce: dos puntos")
            return self.__cruce_dos_puntos()
        elif self.metodo == "uniforme":
            if(self.verbose and self.mark==0):
                print("Metodo de cruce: uniforme")
            return self.__cruce_uniforme()
        else:
            if(self.verbose and self.mark==0):
                print("Metodo de cruce: default")
            return self.__cruce_uniforme()


    # cruce de dos puntos (seleccionamos dos puntos al azar y cruzamos los padres en esos puntos)
    def __cruce_dos_puntos(self):
        # seleccionamos los individuos que pasarán tal cual a la siguiente generación
        [hijos, numero]= self.__individuos_no_modificados()
        
        # realizamos el cruce de dos puntos
        while numero < self.num_ind: 
            # seleccionamos dos indiviuos al azar
            indices = random.sample(range(0, self.padres.shape[0]), 2)
            padre1, padre2 = self.padres[indices]
            
            # seleccionamos 2 puntos aletorios, el primero parte a padre1 y el segundo a padre2
            punto1 = random.randint(0, self.padres.shape[1] - 2)
            punto2 = random.randint(punto1 + 1, self.padres.shape[1] - 1)
            
            # para formar el hijo, se concatena los atributos del padre1 que van desde el inicio al 1 punto, los atributos del padre2
            # que van desde el primer punto al segundo, y los atributos del padre1 que van desde el 2 punto hasta el final
            hijo = np.concatenate((padre1[:punto1], padre2[punto1:punto2], padre1[punto2:]))
            
            # se añade a la lista de hijos
            hijos[numero] = hijo
            numero += 1
        
        return hijos
    

    # cruce uniforme (para cada atributo del hijo, se selecciona el atributo del padre1 o del padre2 aleatoriamente)
    def __cruce_uniforme(self):
        # seleccionamos los individuos que pasarán tal cual a la siguiente generación
        [hijos,numero]= self.__individuos_no_modificados()
        
        # realizamos el cruce uniforme
        while numero < self.num_ind:
            # seleccionamos dos indiviuos al azar
            indices = random.sample(range(0, self.padres.shape[0]), 2)
            padre1, padre2 = self.padres[indices]
            
            # para cada atributo del hijo, se selecciona el atributo del padre1 o del padre2 aleatoriamente
            hijo = np.empty_like(padre1)
            for i in range(len(padre1)):
                hijo[i] = padre1[i] if random.random() < 0.5 else padre2[i]
            
            # se añade a la lista de hijos
            hijos[numero] = hijo
            numero += 1
        
        return hijos
    

    # cruce por defecto (seleccionamos dos padres al azar y cruzamos los padres en un punto al azar)
    def __cruce_default(self):
        # seleccionamos los individuos que pasarán tal cual a la siguiente generación
        [hijos,numero]= self.__individuos_no_modificados()
            
        # realizamos el cruce por defecto
        while numero < self.num_ind:
            # seleccionamos dos padres aletoriamente
            indices = random.sample(range(0, self.padres.shape[0]), 2)
            padre1, padre2 = self.padres[indices]

            # seleccionamos un punto aleatorio que partirá a ambos padres
            punto = random.randint(0, self.padres.shape[1]-1)

            # para formar el hijo, se concatena los atributos del padre1 que van desde el inicio al 1 punto y los 
            # atributos del padre 2 que van desde el punto hasta el final
            hijo = np.concatenate((padre1[:punto], padre2[punto:]))

            # se añade a la lista de hijos
            hijos[numero] = hijo
            numero+=1
            
        return hijos
    

    # método para seleccionar los individuos que no van a ser modificados (el mejor individuo y los que pasan sin cruces)
    def __individuos_no_modificados(self):
        hijos = np.empty_like(self.padres)
        
        # primero, aseguramos que el mejor individuo pasa tal cual
        mejor_individuo = self.padres[0]
        hijos[0] = mejor_individuo
        
        # seleccionamos los individuos que van a pasar sin cruces (-1 por el individuo (el mejor) que ya ha pasado)
        numero = math.ceil(self.num_ind * self.probabilidad_no_cruce) - 1
        indices_sin_cruce = random.sample(range(0, self.padres.shape[0]), numero)
        
        # añadimos el que hemos restado antes para que los métodos funcionen correctamente
        numero += 1
        
        # los añadimos en las n primeras filas de hijos
        seleccionados = self.padres[indices_sin_cruce]
        hijos[1:numero] = seleccionados
        
        return (hijos, numero)
    
    
        