import numpy as np


class Fitness:
    
    def __init__(self, datos, poblacion, nInd):
        self.datos = datos
        self.poblacion = poblacion
        self.nInd = nInd


    # función para calcular el fitness de la población
    def fitness_poblacion(self, dicc_fitness):
        matrix_errors = np.zeros((self.datos.shape[0], self.poblacion.shape[0]))   # matriz de ceros de dimension nDatos x nInd

        # obtenemos para cada ecuacion una lista con el error (diferencia al cuadrado) de cada uno de
        # los individuos de la población, si ya lo habiamos calculado lo obtenemos de dicc_fitness
        for i in range(0, self.datos.shape[0]):
            for j in range(0, self.poblacion.shape[0]):
                individuo = self.poblacion[j, :]
                ecuacion = self.datos.iloc[i]
                clave = (tuple(individuo), tuple(ecuacion))
                if clave in dicc_fitness:
                    error = dicc_fitness[clave]
                else:
                    error = self.__evalua_individuo(ecuacion, individuo)
                    dicc_fitness[clave] = error
                matrix_errors[i, j] = error
        
        # calculamos el fitness de cada individuo como la suma de sus errores en cada una de las ecuaciones
        res = np.zeros(self.nInd)
        for i in range(0, self.nInd):
            res[i] = np.sum(matrix_errors[:, i])   # sumamos los errores de cada ecuacion para el individuo i

        return res


    # función para calcular la diferenca entre el valor real y el valor obtenido para tal individuo
    # en tal ecuacion
    def __evalua_individuo(self, valores, parametros):
        res = 0
        calculos_prohibidos = False     # para evitar indeterminaciones como 0 elevado a 0

        y_real = valores.iloc[-1]
        y_aprox = 0
        
        # calculamos la y_aprox (la ecuacion) para el individuo
        for i in range(0, (parametros.size)-1, 2):    # primero calculamos la suma de los terminos 
            x_i = i // 2        # división entera
            par_i = i
            
            if valores.iloc[x_i] == 0 and parametros[par_i+1] == 0:     # 0 elevado a 0
                y_aprox += 0
                calculos_prohibidos = True
                break
            elif valores.iloc[x_i] == 0 and parametros[par_i+1] < 0:    # division por 0
                y_aprox += 0
                calculos_prohibidos = True
                break
            else:                                                       # calculo normal de la ecuación
                potencia = valores.iloc[x_i] ** parametros[par_i+1]
                y_aprox += parametros[par_i] * potencia

        y_aprox += parametros[-1]   # sumamos el término independiente

        if calculos_prohibidos:
            res = 1000000000000
        else:
            res = (y_real - y_aprox) ** 2   # calculamos el cuadrado de la diferencia

        return res
    

    