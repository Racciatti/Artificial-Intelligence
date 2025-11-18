import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from regressao_logs import regLogs

# Configuração para reprodutibilidade
np.random.seed(42)

def plotar_fronteira_decisao(X, y, modelo, titulo="Fronteira de Decisão"):
    """Plota a fronteira de decisão para dados 2D."""
    plt.figure(figsize=(10, 6))
    
    # Define os limites do gráfico
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Cria grade de pontos
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Calcula probabilidades para cada ponto da grade
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.probabilidade(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Plota contorno
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plota pontos de dados
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', 
                         edgecolors='black', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Classe')
    
    plt.title(titulo)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.show()

def teste_dados_sinteticos():
    """Teste 1: Dados sintéticos simples (2 features)."""
    print("="*70)
    print("TESTE 1: Classificação Binária com Dados Sintéticos")
    print("="*70)
    
    # Gera dados sintéticos
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               random_state=42)
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")
    print(f"Número de features: {X_train.shape[1]}\n")
    
    # Treina o modelo
    modelo = regLogs(learning_rate=0.1, n_iters=1000)
    print("Treinando modelo...")
    modelo.fit(X_train, y_train)
    
    # Faz previsões
    y_pred_train = modelo.predicao(X_train)
    y_pred_test = modelo.predicao(X_test)
    
    # Avalia
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print(f"\nAcurácia no treino: {acc_train:.4f} ({acc_train*100:.2f}%)")
    print(f"Acurácia no teste: {acc_test:.4f} ({acc_test*100:.2f}%)")
    
    print("\nPesos aprendidos:")
    print(f"  Intercepto (bias): {modelo.bias:.4f}")
    print(f"  Pesos das features: {modelo.peso}")
    
    # Matriz de confusão
    print("\nMatriz de Confusão (Teste):")
    print(confusion_matrix(y_test, y_pred_test))
    
    # Plota fronteira de decisão
    plotar_fronteira_decisao(X_test, y_test, modelo, 
                            "Teste 1: Fronteira de Decisão - Dados Sintéticos")

def teste_breast_cancer():
    """Teste 2: Dataset real (Breast Cancer) com todas as features."""
    print("\n" + "="*70)
    print("TESTE 2: Breast Cancer Dataset (Dataset Real)")
    print("="*70)
    
    # Carrega dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Normaliza os dados (importante para convergência)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")
    print(f"Número de features: {X_train.shape[1]}")
    print(f"Classes: {data.target_names}\n")
    
    # Treina o modelo
    modelo = regLogs(learning_rate=0.1, n_iters=2000)
    print("Treinando modelo...")
    modelo.fit(X_train, y_train)
    
    # Faz previsões
    y_pred_train = modelo.predicao(X_train)
    y_pred_test = modelo.predicao(X_test)
    
    # Avalia
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print(f"\nAcurácia no treino: {acc_train:.4f} ({acc_train*100:.2f}%)")
    print(f"Acurácia no teste: {acc_test:.4f} ({acc_test*100:.2f}%)")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_test, target_names=data.target_names))
    
    print("Matriz de Confusão (Teste):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

def teste_probabilidades():
    """Teste 3: Análise de probabilidades e threshold."""
    print("\n" + "="*70)
    print("TESTE 3: Análise de Probabilidades")
    print("="*70)
    
    # Dados simples
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Treina modelo
    modelo = regLogs(learning_rate=0.1, n_iters=1000)
    modelo.fit(X, y)
    
    # Analisa probabilidades
    print("\nPontos de treino e suas probabilidades:")
    print(f"{'X':<15} {'y Real':<10} {'Probabilidade':<15} {'Previsão (0.5)'}")
    print("-" * 60)
    
    probabilidades = modelo.probabilidade(X)
    predicoes = modelo.predicao(X, limiar=0.5)
    
    for i in range(len(X)):
        print(f"{str(X[i]):<15} {y[i]:<10} {probabilidades[i]:<15.4f} {predicoes[i]}")
    
    # Testa diferentes limiares
    print("\nTestando diferentes limiares de decisão:")
    print(f"{'Limiar':<10} {'Previsões'}")
    print("-" * 30)
    
    for limiar in [0.3, 0.5, 0.7]:
        pred = modelo.predicao(X, limiar=limiar)
        print(f"{limiar:<10} {pred}")

def teste_convergencia():
    """Teste 4: Análise de convergência com diferentes learning rates."""
    print("\n" + "="*70)
    print("TESTE 4: Análise de Convergência")
    print("="*70)
    
    # Gera dados
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42)
    
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    print(f"Testando diferentes learning rates:\n")
    
    for lr in learning_rates:
        modelo = regLogs(learning_rate=lr, n_iters=500)
        modelo.fit(X, y)
        y_pred = modelo.predicao(X)
        acc = accuracy_score(y, y_pred)
        
        print(f"Learning Rate: {lr:<6} → Acurácia: {acc:.4f} ({acc*100:.2f}%)")

if __name__ == "__main__":
    # Executa todos os testes
    teste_dados_sinteticos()
    teste_breast_cancer()
    teste_probabilidades()
    teste_convergencia()
    
    print("\n" + "="*70)
    print("TODOS OS TESTES CONCLUÍDOS!")
    print("="*70)
