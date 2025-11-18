import numpy as np
from typing import Optional

class KMeans:

    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-4):
        self.k = k # número de clusters
        self.max_iters = max_iters
        self.tol = tol #tolerancia para convergência
        self.centroids: Optional[np.ndarray] = None
        self.rotulo: Optional[np.ndarray] = None # rótulos de cluster para cada ponto

    def startCentroids(self, N: np.ndarray) -> np.ndarray:
        amostras = N.shape[0]
        ind_aleatorios = np.random.choice(amostras, size=self.k, replace=False) # Seleciona k índices únicos aleatoriamente
        centroides_iniciais = N[ind_aleatorios] # Os centroides iniciais
        return centroides_iniciais
    
    def distancia_euclidiana(self, pt1: np.ndarray, pt2: np.ndarray) -> float:
        return np.sum((pt1 - pt2) ** 2) # Distância Euclidiana ao quadrado
    
    def atribuir_clusters(self, N: np.ndarray) -> np.ndarray:
        n_amostras = N.shape[0]
        rotulo = np.zeros(n_amostras, dtype=int) # Inicializa rótulos de cluster

        for i, ponto in enumerate(N):
            min_dist = float('inf') # Começa com "infinito"
            ind_centroide_proximo = -1


            #descobrindo o centroide mais próximo
            for k in range(self.k):
                dist = self.distancia_euclidiana(ponto, self.centroids[k])
                if dist < min_dist:
                    min_dist = dist
                    ind_centroide_proximo = k 

            rotulo[i] = ind_centroide_proximo

        return rotulo
    
    def uptd_centroide(self, N: np.ndarray, rotulo: np.ndarray) -> np.ndarray:
        novoCentroids = np.zeros((self.k, N.shape[1])) # Inicializa novos centroides, ou seja, reescrevo por cima. Tem k linhas e n colunas

        for k in range(self.k):
            pCluester = N[rotulo == k] # Seleciona todos os pontos que pertencem ao cluster k

            if len(pCluester) > 0:
                novoCentroids[k] = np.mean(pCluester, axis=0)
            else:
                novoCentroids[k] = self.centroids[k] # mantém o centroide K antigo

        return novoCentroids
    
    def fit (self, N: np.ndarray):
        N = np.asarray(N)
        self.centroids = self.startCentroids(N) # Inicializa os centroides

        for i in range(self.max_iters):
            oldCentroid = self.centroids.copy()
            self.rotulo = self.atribuir_clusters(N)
            self.centroids = self.uptd_centroide(N, self.rotulo)

            # Verifica convergência
            diff = np.sum(np.linalg.norm(self.centroids - oldCentroid, axis=1)) 

            if diff < self.tol:
                print(f'Convergido após {i+1} iterações.')
                break

            print(f'Iteração {i+1}, diferença dos centroides: {diff}')

        self.rotulo = self.atribuir_clusters(N)
        print("K-Means finalizado.")

    def predict(self, N: np.ndarray) -> np.ndarray:
        N = np.asarray(N)
        
        if self.centroids is None:
            raise Exception("O modelo KMeans não foi treinado. Chame o método 'fit' primeiro.")
        
        return self.atribuir_clusters(N)
    
    def get_centroids(self) -> Optional[np.ndarray]:
        if self.centroids is None:
            raise Exception("O modelo KMeans não foi treinado. Chame o método 'fit' primeiro.")
        return self.centroids

        
        