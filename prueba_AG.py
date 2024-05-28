import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
import time

# La siguiente linea es un ejemplo
# Modificar los import como sea necesario para cargar el AG implementado
from AG_JMoyano import AG

# Nombre generico del dataset
nombre_dataset = 'toy1'

nombre_dataset_train = nombre_dataset+"_train.csv"
nombre_dataset_val = nombre_dataset+"_val.csv"

# La clase AG debe estar implementada
# (importe los ficheros necesarios antes de ejecutar las siguientes lineas)
ag = AG(
	# datos de entrenamiento (para el proceso del AG)
	datos_train = nombre_dataset_train, 
	# datos de validacion/test (para predecir)
	datos_test = nombre_dataset_val, 
	# semilla para numeros aleatorios
	seed=123, 
	# numero de individuos
	nInd = 50, 
	# maximo de iteraciones
	maxIter = 100 
)

# Ejecucion del AG midiendo el tiempo
inicio = time.time()
ind, y_pred = ag.run()
fin = time.time()
print(f'Tiempo ejecucion: {(fin-inicio):.2f}')

# Imprimir mejor solución encontrada
print(f'Mejor individuo: {ind}')
# 0.5*(a1^2) + -0.3*(a2^1) + ... + 10 # --> Se trata de un ejemplo

# Imprimir predicciones sobre el conjunto de test
print(f'Predicciones: {y_pred}')
#  [-1.53, 1.49, 2.15, ..., -2.77] # --> Se trata de un ejemplo

# Cargar valores reales de 'y' en el conjunto de validacion/test 
#   y calcular RMSE y R2 con las predicciones del AG
y_true = pd.read_csv(nombre_dataset_val)['y']
rmse = root_mean_squared_error(y_true, y_pred)
print(f'RMSE: {rmse:.4f}')

r2 = r2_score(y_true, y_pred)
print(f'R2: {r2:.4f}')