import pandas as pd
from src.AG1_LinaresBarrera_ChicoCastellano import AG1
from sklearn.metrics import mean_squared_error, r2_score
import ast

class GeneticAlgorithmTesterFiltered:
    
    #Es una clase muy similar al de test, solo que en vez de probar con todas las combinaciones de hiperparámetros, solo utiliza las n primeras del dataset que genera la otra clase test
    #El pbjetivo es poder probar las mejores combinaciones de hiperparámetros en varios ficheros, y desechar aquellas que no hayan funcionado correctamente
    def __init__(self, param_combinations, datos_train, datos_test, seed, num_ind, max_iter, verbose):
        self.param_combinations = param_combinations
        self.results = []
        self.datos_train = datos_train
        self.datos_test = datos_test
        self.seed = seed
        self.num_ind = num_ind
        self.max_iter = max_iter
        self.verbose = verbose

    #Ejecutamos el algoritmo 20 veces y calculamos la media de r2 para saber su rendimiento
    def AG(self, population_method, crossover_method, probabilidad_baja, probabilidad_alta, k, tasa_elitismo, tasa_no_cruce, combo):
        r2_scores = []
        for _ in range(20):
            ag = AG1(
                datos_train=self.datos_train, 
                datos_test=self.datos_test, 
                seed=self.seed, 
                num_ind=self.num_ind, 
                max_iter=self.max_iter,
                verbose=self.verbose,
                population_method=population_method,
                crossover_method=crossover_method,
                probabilidad_baja=probabilidad_baja, 
                probabilidad_alta=probabilidad_alta, 
                k=k, 
                tasa_elitismo=tasa_elitismo, 
                tasa_no_cruce=tasa_no_cruce,
            )

            ind, y_pred = ag.run()

            y_true = pd.read_csv(self.datos_test)['y']
            r2 = r2_score(y_true, y_pred)
            r2_scores.append(r2)
            
        avg_r2 = sum(r2_scores) / len(r2_scores)
        print(f'Promedio R2: {avg_r2:.4f}. Combinacion: {combo}')
        
        return avg_r2

    #Para als n primeras combinaciónes de hiperparámetros llamamos al método que ejecuta el algoritmo
    def run_tests(self):
        print("Comprobando hiperparametros:")
        for combo in self.param_combinations:
            population_method, crossover_method, probabilidad_baja, probabilidad_alta, k, tasa_elitismo, tasa_no_cruce = combo
            resultado = self.AG(
                population_method=population_method,
                crossover_method=crossover_method,
                probabilidad_baja=probabilidad_baja,
                probabilidad_alta=probabilidad_alta,
                k=k,
                tasa_elitismo=tasa_elitismo,
                tasa_no_cruce=tasa_no_cruce,
                combo=combo
            )
            self.results.append((combo, resultado))

        #Ordenamos los resultados de mayor a menor en función de la media de R2
        self.results.sort(key=lambda x: x[1], reverse=True)
        mejor_combinacion, mejor_resultado = self.results[0]
        
        #Printeamos la mejor combinación con el mejor resultado
        print(f"Mejor combinación: {mejor_combinacion} con resultado: {mejor_resultado}")
        self.save_results()
        return mejor_combinacion, mejor_resultado

    #Metodo que guarda en un csv los resultados obtenidos por el algoritmo
    def save_results(self):
        df = pd.DataFrame(self.results, columns=['Hiperparámetros', 'R2'])
        df.to_csv('data/resultados_algoritmo_genetico_filtrado.csv', index=False)
        print("Resultados guardados en 'resultados_algoritmo_genetico_filtrado.csv'")


#Utilización
#Lee el fichero que le proporcionemos saltado la primera línea (Es donde se pone los nombres de las columnas)
filename = 'data/resultados_algoritmo_genetico_filtrado_housing.csv' 
df = pd.read_csv(filename, header=None, skiprows=1, encoding='utf-8')

# Seleccionar las 10 primeras combinaciones
filtered_combinations = [ast.literal_eval(combo[0]) for combo in df.values[:20]]

nombre_dataset = 'synt1'

nombre_dataset_train = "data/" + nombre_dataset + "_train.csv"
nombre_dataset_val = "data/" + nombre_dataset + "_val.csv"

tester = GeneticAlgorithmTesterFiltered(
    param_combinations = filtered_combinations,
    datos_train = nombre_dataset_train, 
    datos_test = nombre_dataset_val, 
    seed = 123, 
    num_ind = 100, 
    max_iter = 500,
    verbose = False
)

mejor_combinacion, mejor_resultado = tester.run_tests()
