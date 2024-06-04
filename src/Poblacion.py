import numpy as np

class Poblacion:
    def __init__(self, nInd, nAtrib,verbose, method):
        self.nInd = nInd
        self.nAtrib = nAtrib
        self.method = method
        self.verbose = verbose

    # función para crear la población inicial con estrategias mejoradas
    def initial(self):
        if self.method == "diverse":
            if(self.verbose):
                print("Metodo de generacion de poblacion inicial: diverse")
            return self._initial_diverse()
        elif self.method == "seeded":
            if(self.verbose):
                print("Metodo de generacion de poblacion inicial: seeded")
            return self._initial_seeded()
        else:
            if(self.verbose):
                print("Metodo de generacion de poblacion inicial: default")
            return self._initial_default()

    def _initial_default(self):
        poblacion_inicial = np.zeros((self.nInd, 2 * self.nAtrib + 1))

        # números decimales aleatorios para las posiciones pares entre -50 y 50
        indices_pares = range(0, 2 * self.nAtrib, 2)
        poblacion_inicial[:, indices_pares] = np.random.uniform(-50, 50, size=(self.nInd, self.nAtrib))
        
        # números enteros aleatorios para las posiciones impares (excepto la última) entre -1 y 5
        indices_impares = range(1, 2 * self.nAtrib, 2)
        poblacion_inicial[:, indices_impares] = np.random.randint(-1, 5, size=(self.nInd, self.nAtrib))
        
        # números decimales aleatorio en el rango para la última posición impar entre -50 y 50
        poblacion_inicial[:, 2*self.nAtrib] = np.random.uniform(-50, 50, size=self.nInd)

        return poblacion_inicial


    def _initial_diverse(self):
        poblacion_inicial = np.zeros((self.nInd, 2 * self.nAtrib + 1))
        
        # Número de clusters (grupos de soluciones)
        num_clusters = 5
        
        # Calcula el tamaño de cada cluster (número de individuos por cluster)
        cluster_size = self.nInd // num_clusters
        
        # Itera sobre el número de clusters
        for i in range(num_clusters):
            
            # Genera el centro del cluster utilizando un número aleatorio entre -50 y 50 
            # Este será el punto central alrededor del cual se generarán los individuos del cluster
            cluster_center = np.random.uniform(-50, 50, size=(1, 2 * self.nAtrib + 1))
            
            # Itera sobre el número de individuos dentro de cada cluster
            for j in range(cluster_size):
                
                # Para las posiciones pares, se añade una perturbación normal (distribución normal con media 0 y desviación estándar 10) al valor del centro del cluster.
                indices_pares = range(0, 2 * self.nAtrib, 2)
                poblacion_inicial[i * cluster_size + j, indices_pares] = ( cluster_center[:, indices_pares] + np.random.normal(0, 10, size=(1, self.nAtrib)))
                
                # Para las posiciones impares, se selecciona aletoriamente un valor entre -1 y 5
                indices_impares = range(1, 2 * self.nAtrib, 2)
                poblacion_inicial[i * cluster_size + j, indices_impares] =  np.random.randint(-1, 5, size=(1, self.nAtrib))
                
                # Para el término independiente, se añade una modificación normal (distribución normal con media 0 y desviación estándar 10) al valor del centro del cluster.
                poblacion_inicial[i * cluster_size + j, 2 * self.nAtrib] = ( cluster_center[:, 2 * self.nAtrib] + np.random.normal(0, 10))
        
        return poblacion_inicial


    def _initial_seeded(self):
        poblacion_inicial = np.zeros((self.nInd, 2 * self.nAtrib + 1))
        seed_solutions = np.array([
            [2.19314048, 3, 8.47209093, 0, 2.8537148, 1, -1.02032502, 5, 4.06379636, 0, -12.56803537]
            
        ])
        seed_size = len(seed_solutions)
        for i in range(seed_size):
            poblacion_inicial[i] = seed_solutions[i]
        for i in range(seed_size, self.nInd):
            
            indices_pares = range(0, 2 * self.nAtrib, 2)
            poblacion_inicial[i, indices_pares] = np.random.uniform(-50, 50, size=(1, self.nAtrib))
            
            indices_impares = range(1, 2 * self.nAtrib, 2)
            poblacion_inicial[i, indices_impares] = np.random.randint(-1, 5, size=(1, self.nAtrib))
            
            poblacion_inicial[i, 2 * self.nAtrib] = np.random.uniform(-50, 50)
            
        print(poblacion_inicial)
            
    
        return poblacion_inicial
