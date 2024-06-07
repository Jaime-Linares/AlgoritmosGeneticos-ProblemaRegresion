class Prediccion:   

    def __init__(self, datos_prueba, individuo_solucion):
        self.datos_prueba = datos_prueba
        self.individuo_solucion = individuo_solucion


    # método que devuelve las predicciones del individuo solución para el dataset de prueba
    def predecir(self):
        prediccion = []

        for j in range(0, self.datos_prueba.shape[0]):
            y_aprox=0
            valores = self.datos_prueba.iloc[j, :-1]
            for i in range(0, self.individuo_solucion.size-1, 2):
                x_i = i // 2        # división entera
                par_i = i
                if valores.iloc[x_i] == 0 and self.individuo_solucion[par_i+1] == 0:     # 0 elevado a 0
                    y_aprox += 1000000000000
                elif valores.iloc[x_i] == 0 and self.individuo_solucion[par_i+1] < 0:    # division por 0
                    y_aprox += 1000000000000
                else:                                                                    # calculo normal de la ecuación
                    potencia = valores.iloc[x_i] ** self.individuo_solucion[par_i+1]
                    y_aprox += self.individuo_solucion[par_i] * potencia
            y_aprox += self.individuo_solucion[-1]   # sumamos el término independiente
            prediccion.append(y_aprox)
        
        return prediccion
    

    