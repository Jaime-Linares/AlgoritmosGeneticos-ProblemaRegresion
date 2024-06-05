import numpy as np
import random
import math

class Cruce:


    def __init__(self, padres, numero_individuos, probabilidad_no_cruce, fitness, metodo, marca, verbose):

        self.padres = padres
        self.numero_individuos = numero_individuos
        self.probabilidad_no_cruce = probabilidad_no_cruce
        self.fitness = fitness
        self.metodo = metodo
        self.marca = marca
        self.verbose= verbose

    def cruzar(self):
        if self.metodo == "dos_puntos":
            if(self.verbose and self.marca==0):
                print("Metodo de cruce: Dos puntos")
            return self._cruce_dos_puntos()
        elif self.metodo == "uniforme":
            if(self.verbose and self.marca==0):
                print("Metodo de cruce: Uniforme")
            return self._cruce_uniforme()
        else:
            if(self.verbose and self.marca==0):
                print("Metodo de cruce: Default")
            return self._cruce_uniforme()

    def _cruce_dos_puntos(self):
        
        #Seleccionamos los individuos que pasarán tal cual a la siguiente generación
        [hijos,numero]= self._individuos_no_modificados()
        
        #Cruce de dos puntos
        while numero < self.numero_individuos:
            
            #Seleccionamos dos indiviuos al azar
            indices = random.sample(range(0, self.padres.shape[0]), 2)
            padre1, padre2 = self.padres[indices]
            
            #Seleccionamos 2 puntos aletorios, el primero parte a padre1 y el segundo a padre2
            punto1 = random.randint(0, self.padres.shape[1] - 2)
            punto2 = random.randint(punto1 + 1, self.padres.shape[1] - 1)
            
            #Para formar el hijo, se concatena los atributos del padre1 que van desde el inicio al 1 punto, los atributos del padre2 que van desde
            #el primer punto al segundo, y los atributos del padre1 que van desde el 2 punto hasta el final
            hijo = np.concatenate((padre1[:punto1], padre2[punto1:punto2], padre1[punto2:]))
            
            #Se añade a la lista de hijos
            hijos[numero] = hijo
            numero += 1
        
        return hijos

    def _cruce_uniforme(self):
        
        #Seleccionamos los individuos que pasarán tal cual a la siguiente generación
        [hijos,numero]= self._individuos_no_modificados()
        
        # Cruce uniforme
        while numero < self.numero_individuos:
            
            #Seleccionamos dos indiviuos al azar
            indices = random.sample(range(0, self.padres.shape[0]), 2)
            padre1, padre2 = self.padres[indices]
            
            #Para cada atributo del hijo, se selecciona el atributo del padre1 o del padre2 aleatoriamente
            hijo = np.empty_like(padre1)
            for i in range(len(padre1)):
                hijo[i] = padre1[i] if random.random() < 0.5 else padre2[i]
            
            #Se añade a la lista de hijos
            hijos[numero] = hijo
            numero += 1
        
        return hijos

    def _cruce_default(self):
        
        #Seleccionamos los individuos que pasarán tal cual a la siguiente generación
        [hijos,numero]= self._individuos_no_modificados()
            
        #Para el resto de individuos cruzamos aleatoriamente dos de los padres
        while numero < self.numero_individuos:

            #Seleccionamos dos padres aletoriamente
            indices = random.sample(range(0, self.padres.shape[0]), 2)
            padre1, padre2 = self.padres[indices]

            #Seleccionamos seleccionamos un punto aleatorio que partirá a ambos padres
            punto = random.randint(0, self.padres.shape[1]-1)

            #Para formar el hijo, se concatena los atributos del padre1 que van desde el inicio al 1 punto y los 
            #atributos del padre 2 que van desde el punto hasta el final
            hijo = np.concatenate((padre1[:punto], padre2[punto:]))

            #Se añade a la lista de hijos
            hijos[numero] = hijo
            numero+=1
            
        return hijos
    
    def _individuos_no_modificados(self):
        hijos = np.empty_like(self.padres)
        
        #Primero, aseguramos que el mejor individuo pasa tal cual
        mejor_individuo = self.padres[0]
        hijos[0] = mejor_individuo
        
        #Seleccionamos los individuos que van a pasar sin cruces (-1 por el individuo que ya ha pasado)
        numero = math.ceil(self.numero_individuos * self.probabilidad_no_cruce) - 1
        indices_sin_cruce = random.sample(range(0, self.padres.shape[0]), numero)
        
        #Añadimos el que hemos restado antes para que los métodos funcionen correctamente
        numero+=1
        
        #los añadimos en las n primeras filas de hijos
        seleccionados = self.padres[indices_sin_cruce]
        hijos[1:numero] = seleccionados
        
        return hijos, numero
        