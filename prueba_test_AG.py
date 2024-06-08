import itertools
import pandas as pd
from src.AG1_LinaresBarrea_ChicoCastellano import AG1
from sklearn.metrics import root_mean_squared_error, r2_score
import time

class GeneticAlgorithmTester:
    
    def __init__(self, population_method_vals, crossover_method_vals, probabilidad_baja_vals, probabilidad_alta_vals, k_vals, tasa_elitismo_vals, tasa_no_cruce_vals, datos_train, datos_test, seed, num_ind, max_iter, verbose):
        self.param_grid = {
            'population_method': population_method_vals,
            'crossover_method': crossover_method_vals,
            'probabilidad_baja': probabilidad_baja_vals,
            'probabilidad_alta': probabilidad_alta_vals,
            'k': k_vals,
            'tasa_elitismo': tasa_elitismo_vals,
            'tasa_no_cruce': tasa_no_cruce_vals
        }
        self.param_combinations = list(itertools.product(
            self.param_grid['population_method'],
            self.param_grid['crossover_method'],
            self.param_grid['probabilidad_baja'],
            self.param_grid['probabilidad_alta'],
            self.param_grid['k'],
            self.param_grid['tasa_elitismo'],
            self.param_grid['tasa_no_cruce']
        ))
        self.results = []
        self.datos_train = datos_train
        self.datos_test = datos_test
        self.seed = seed
        self.num_ind = num_ind
        self.max_iter = max_iter
        self.verbose = verbose

    def AG(self, population_method, crossover_method,probabilidad_baja, probabilidad_alta, k, tasa_elitismo, tasa_no_cruce, combo):
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

            ind,y_pred = ag.run()

            y_true = pd.read_csv(self.datos_test)['y']
            r2 = r2_score(y_true, y_pred)
            r2_scores.append(r2)
            
        avg_r2 = sum(r2_scores) / len(r2_scores)
        print(f'Promedio R2: {avg_r2:.4f}. Combinacion: {combo}')
            
            

        return avg_r2

    def run_tests(self):
        print("Comprobando hiperparametros:")
        for combo in self.param_combinations:
            population_method,crossover_method,probabilidad_baja, probabilidad_alta, k, tasa_elitismo, tasa_no_cruce = combo
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

        self.results.sort(key=lambda x: x[1],reverse=True)
        mejor_combinacion, mejor_resultado = self.results[0]
        print(f"Mejor combinación: {mejor_combinacion} con resultado: {mejor_resultado}")
        self.save_results()
        return mejor_combinacion, mejor_resultado

    def save_results(self):
        df = pd.DataFrame(self.results, columns=['Hiperparámetros', 'RMSE'])
        df.to_csv('data/resultados_algoritmo_genetico.csv', index=False)
        print("Resultados guardados en 'resultados_algoritmo_genetico.csv'")

# Ejemplo de uso
nombre_dataset = 'toy1'

nombre_dataset_train = "data/" + nombre_dataset + "_train.csv"
nombre_dataset_val = "data/" + nombre_dataset + "_val.csv"


tester = GeneticAlgorithmTester(
	datos_train = nombre_dataset_train, 
	datos_test = nombre_dataset_val, 
	seed = 123, 
	num_ind = 100, 
	max_iter = 500,
	verbose = False,
	population_method_vals = ["diverse", "default"],
	crossover_method_vals = ["dos_puntos", "uniforme", "default"],
    probabilidad_baja_vals=[0.01, 0.05, 0.1, 0.2],
    probabilidad_alta_vals=[0.6, 0.7, 0.8, 0.9],
    k_vals=[2, 3, 5, 7],
    tasa_elitismo_vals=[0.01, 0.05, 0.1, 0.2],
    tasa_no_cruce_vals=[0.2, 0.4, 0.6, 0.8],
)

mejor_combinacion, mejor_resultado = tester.run_tests()
