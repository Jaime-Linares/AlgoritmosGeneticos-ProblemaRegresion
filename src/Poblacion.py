import numpy as np


class Poblacion:
    def __init__(self, nInd, nAtrib,verbose, method, dataset):
        self.nInd = nInd
        self.nAtrib = nAtrib
        self.method = method
        self.verbose = verbose
        self.dataset = dataset


    # función para crear la población inicial con estrategias mejoradas
    def initial(self):
        if self.method == "diverse":
            if(self.verbose):
                print("Metodo de generacion de poblacion inicial: diverse")
            return self.__initial_diverse()
        elif self.method == "seeded":
            if(self.verbose):
                print("Metodo de generacion de poblacion inicial: seeded")
            return self.__initial_seeded()
        else:
            if(self.verbose):
                print("Metodo de generacion de poblacion inicial: default")
            return self.__initial_default()


    # método para generar la población inicial por defecto
    def __initial_default(self):
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


    # método para generar la población inicial con diversidad
    def __initial_diverse(self):
        poblacion_inicial = np.zeros((self.nInd, 2 * self.nAtrib + 1))
        
        # número de clusters (grupos de soluciones)
        num_clusters = 5
        
        # calcula el tamaño de cada cluster (número de individuos por cluster)
        cluster_size = self.nInd // num_clusters
        
        # itera sobre el número de clusters
        for i in range(num_clusters):          
            # genera el centro del cluster utilizando un número aleatorio entre -50 y 50 
            # este será el punto central alrededor del cual se generarán los individuos del cluster
            cluster_center = np.random.uniform(-50, 50, size=(1, 2 * self.nAtrib + 1))
            
            # itera sobre el número de individuos dentro de cada cluster
            for j in range(cluster_size):                
                # para las posiciones pares, se añade una perturbación normal (distribución normal con media 0 y desviación estándar 10) al valor del centro del cluster.
                indices_pares = range(0, 2 * self.nAtrib, 2)
                poblacion_inicial[i * cluster_size + j, indices_pares] = ( cluster_center[:, indices_pares] + np.random.normal(0, 10, size=(1, self.nAtrib)))
                
                # para las posiciones impares, se selecciona aletoriamente un valor entre -1 y 5
                indices_impares = range(1, 2 * self.nAtrib, 2)
                poblacion_inicial[i * cluster_size + j, indices_impares] =  np.random.randint(-1, 5, size=(1, self.nAtrib))
                
                # para el término independiente, se añade una modificación normal (distribución normal con media 0 y desviación estándar 10) al valor del centro del cluster.
                poblacion_inicial[i * cluster_size + j, 2 * self.nAtrib] = ( cluster_center[:, 2 * self.nAtrib] + np.random.normal(0, 10))
        
        return poblacion_inicial


    # método para generar la población inicial con semillas
    def __initial_seeded(self):
        poblacion_inicial = np.zeros((self.nInd, 2 * self.nAtrib + 1))

        if (self.dataset== "data/toy1_train.csv"):
            seed_solutions = np.array([
                [1.65435962, 3, -1.19174116, 3, 2.15330594, 1, -0.34445033, 5, -3.41407396, 0, 3.69980822],
                [1.65435962, 3, -1.19174116, 3, 2.07242115, 1, -0.34445033, 1, -3.41407396, 0, 3.69980822]
            ])
        elif (self.dataset== "data/synt1_train.csv"):
            seed_solutions = np.array([

            ])
        elif (self.dataset== "data/housing_train.csv"):
            seed_solutions = np.array([

            ])
        else:
            seed_solutions = np.array([
                
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
    
        return poblacion_inicial
    

