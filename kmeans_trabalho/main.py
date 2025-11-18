import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans

# Configuração para reprodutibilidade
np.random.seed(42)

def gerar_dados_teste():
    """Gera dados sintéticos com 3 clusters distintos."""
    # Cluster 1: centrado em (2, 2)
    cluster1 = np.random.randn(50, 2) * 0.5 + np.array([2, 2])
    
    # Cluster 2: centrado em (8, 3)
    cluster2 = np.random.randn(50, 2) * 0.5 + np.array([8, 3])
    
    # Cluster 3: centrado em (5, 8)
    cluster3 = np.random.randn(50, 2) * 0.5 + np.array([5, 8])
    
    # Combina todos os clusters
    X = np.vstack([cluster1, cluster2, cluster3])
    
    return X

def plotar_resultados(X, kmeans, titulo="K-Means Clustering"):
    """Plota os dados e os centroides encontrados."""
    plt.figure(figsize=(10, 6))
    
    # Plota os pontos coloridos por cluster
    cores = ['red', 'blue', 'green', 'orange', 'purple']
    for k in range(kmeans.k):
        pontos_cluster = X[kmeans.rotulo == k]
        plt.scatter(pontos_cluster[:, 0], pontos_cluster[:, 1], 
                   c=cores[k % len(cores)], label=f'Cluster {k}', 
                   alpha=0.6, s=50)
    
    # Plota os centroides
    centroides = kmeans.get_centroids()
    plt.scatter(centroides[:, 0], centroides[:, 1], 
               c='black', marker='X', s=300, 
               edgecolors='yellow', linewidths=2,
               label='Centroides')
    
    plt.title(titulo)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def teste_basico():
    """Teste básico do K-Means com dados sintéticos."""
    print("="*60)
    print("TESTE 1: K-Means com dados sintéticos (k=3)")
    print("="*60)
    
    # Gera dados
    X = gerar_dados_teste()
    print(f"Dataset gerado: {X.shape[0]} amostras, {X.shape[1]} features\n")
    
    # Treina K-Means
    kmeans = KMeans(k=3, max_iters=100, tol=1e-4)
    kmeans.fit(X)
    
    # Mostra resultados
    print(f"\nCentroides finais:")
    centroides = kmeans.get_centroids()
    for i, centroide in enumerate(centroides):
        print(f"  Cluster {i}: {centroide}")
    
    print(f"\nDistribuição de pontos por cluster:")
    for k in range(kmeans.k):
        count = np.sum(kmeans.rotulo == k)
        print(f"  Cluster {k}: {count} pontos")
    
    # Plota
    plotar_resultados(X, kmeans, "Teste 1: K-Means com k=3")

def teste_predict():
    """Teste do método predict com novos pontos."""
    print("\n" + "="*60)
    print("TESTE 2: Previsão de novos pontos")
    print("="*60)
    
    # Gera e treina
    X = gerar_dados_teste()
    kmeans = KMeans(k=3, max_iters=100, tol=1e-4)
    kmeans.fit(X)
    
    # Novos pontos para classificar
    novos_pontos = np.array([
        [2.5, 2.5],  # Deve ir para cluster próximo de (2, 2)
        [8.0, 3.0],  # Deve ir para cluster próximo de (8, 3)
        [5.0, 8.0],  # Deve ir para cluster próximo de (5, 8)
    ])
    
    print("\nClassificando novos pontos:")
    predicoes = kmeans.predict(novos_pontos)
    
    for i, (ponto, cluster) in enumerate(zip(novos_pontos, predicoes)):
        print(f"  Ponto {ponto} → Cluster {cluster}")

def teste_diferentes_k():
    """Teste com diferentes valores de k."""
    print("\n" + "="*60)
    print("TESTE 3: Comparando diferentes valores de k")
    print("="*60)
    
    X = gerar_dados_teste()
    
    for k in [2, 3, 4, 5]:
        print(f"\n--- Testando k={k} ---")
        kmeans = KMeans(k=k, max_iters=50, tol=1e-4)
        kmeans.fit(X)
        
        print(f"Distribuição de pontos:")
        for cluster_id in range(k):
            count = np.sum(kmeans.rotulo == cluster_id)
            print(f"  Cluster {cluster_id}: {count} pontos")

def teste_dados_simples():
    """Teste com dados muito simples para validação visual."""
    print("\n" + "="*60)
    print("TESTE 4: Dados simples (2 clusters óbvios)")
    print("="*60)
    
    # Dados simples: 2 grupos bem separados
    X = np.array([
        [1, 1], [1.5, 2], [1, 0.6],    # Grupo 1
        [8, 8], [9, 9], [8.5, 8.2]     # Grupo 2
    ])
    
    print(f"Dataset: {X}")
    
    kmeans = KMeans(k=2, max_iters=100, tol=1e-4)
    kmeans.fit(X)
    
    print(f"\nCentroides encontrados:")
    centroides = kmeans.get_centroids()
    for i, centroide in enumerate(centroides):
        print(f"  Cluster {i}: {centroide}")
    
    print(f"\nRótulos dos pontos: {kmeans.rotulo}")

if __name__ == "__main__":
    # Executa todos os testes
    teste_basico()
    teste_predict()
    teste_diferentes_k()
    teste_dados_simples()
    
    print("\n" + "="*60)
    print("TODOS OS TESTES CONCLUÍDOS!")
    print("="*60)
