# Gerar possíveis pontos de corte
# – Para atributos numéricos: usar limites entre valores ordenados.
# – Para categóricos: dividir entre “categoria == X” e “categoria ≠ X” (versão simples).

# Particionar o dataset em:

# lado esquerdo (≤ threshold)

# lado direito (> threshold)

# Calcular o Gini de cada lado.

# Calcular o Gini ponderado do split.

# Escolher o split que gera a menor impureza.

# threshold = {v_i + v_{i+1}}/2 
# Utiliza a média entre valores consecutivos

import numpy as np

def calcular_gini(y):
    """
    Função para calcular o Gini de um vetor de rótulos, características, classes que temos no conjunto
    Gini = 1 - somatório das proporções ao quadrado

    Parameters 
    ----------
    y: 
        Vetor de rótulos com as classes

    Returns
    -------
    gini: float
        Valor do gini daquela classe. Indica a "impureza" dos dados, ou seja, se os dados ainda podem ser separados.

    """
    # proporção é a quantidade de exemplos da classe i pelo total de exemplos 
    total = len(y)
    proporcao = {}

    for caracteristica in y: 
        proporcao[caracteristica] = proporcao.get(caracteristica, 0) + 1
    
    for classe in proporcao:
        proporcao[classe] /= total


    gini = 1 - sum(p**2 for p in proporcao.values())

    return gini

def dividir_dataset(X, y, atributo, threshold): 
    """
    Receber um atributo e atribuir um ponto de corte, isto é, retornar um grupo a direita e a esquerda do threshold 

    Parameters 
    ----------
    X: np.array
        Matriz de atributos
    y: 
        rótulos 
    atributo:
        índice da coluna 
    threshold: 
        ponto de corte; valor usado para dividir a função em duas


    Returns 
    -------
    X_esq: list
        subconjunto de atributos do lado esquerdo do split
    y_esq: list
            Rótulos que estão do lado ESQUERDO do split
    X_dir: list 
        Subconjunto de atributos do lado DIREITO do split
    y_dir: list 
        Rótulos que estão do lado DIREITO do split

    """

    # 1° passo - criar listas para dividir em dois grupos
    X_esq = []
    X_dir = []
    y_esq = []
    y_dir = []


    # 2° passo - percorrer o dataset
    for i in range(len(X)):
        # se for numérico 
        if X[i][atributo] <= threshold: 
            X_esq.append(X[i])
            y_esq.append(y[i])
        else:
            X_dir.append(X[i])
            y_dir.append(y[i])

        # Se for categório
        # if X[i][atributo] == threshold:
        #     X_esq.append(X[i])
        # else:
        #     X_dir.append(X[i])

    return X_esq, y_esq, X_dir, y_dir

def calcular_gini_split(y_esq, y_dir):
    # 1° passo - calcular o gini de cada grupo    
    gini_esq = calcular_gini(y_esq)
    gini_dir = calcular_gini(y_dir)

    # 2° passo - calcular o tamanho (para usar na fórmula depois)
    tam_esq = len(y_esq)
    tam_dir = len(y_dir) 
    tam_total = tam_esq + tam_dir

    # 3° passo - gini ponderaod 
    # esq/total * gini da esquerda + dir/total gini da direita
    gini_split = (tam_esq/tam_total) * gini_esq + (tam_dir/tam_total) * gini_dir

    # quanto menor o gini melhor
    return gini_split

    

def melhor_split(X, y):
    """
    Entre todos os atributos e possíveis valores de corte, encontrar 
    - melhor atributo; 
    - thresholds; 
    - melhor valor;
    - menor valor do gini separado

    Parameters
    ----------
    X: np.array
        Matriz de atributos 
    y: 
        Vetor de rótulos correspondente a CADA linha de X

    Returns
    -------
    melhor_atributo: int
        Índice do melhor atributo para fazer o split
    melhor_threshold: float
        Melhor valor de corte para dividir o dataset
    melhor_gini: float
        Menor valor de Gini encontrado

    """
    # 0° passo - inicializar variáveis para armazenar o melhor split
    melhor_atributo = None 
    melhor_threshold = None
    melhor_gini = float("inf")
    melhor_X_esq = None
    melhor_y_esq = None
    melhor_X_dir = None
    melhor_y_dir = None
    
    
    # Número de atributos (colunas)
    num_atributos = X.shape[1]
    
    # 1° passo - iterar sobre CADA atributo
    for atributo in range(num_atributos):
        # Extrair os valores deste atributo
        valores_atributo = X[:, atributo]
        
        # Ordenar os valores únicos para encontrar os thresholds
        valores_unicos = np.unique(valores_atributo)
        
        # 2° passo - encontrar os thresholds possíveis para este atributo
        for i in range(len(valores_unicos) - 1):
            # Calcular threshold como média entre valores consecutivos
            threshold_atual = (valores_unicos[i] + valores_unicos[i+1]) / 2
            
            # 3° passo - dividir o dataset usando este atributo e threshold
            X_esq, y_esq, X_dir, y_dir = dividir_dataset(X, y, atributo, threshold_atual)
            
            # Verificar se o split é válido (ambos os lados têm dados)
            if len(y_esq) == 0 or len(y_dir) == 0:
                # Split não divide os dados, é inútil
                continue
            
            # 4° passo - calcular o Gini deste split
            gini_atual = calcular_gini_split(y_esq, y_dir)
            
            # 5° passo - atualizar se encontramos um split melhor
            if gini_atual < melhor_gini:
                melhor_gini = gini_atual
                melhor_atributo = atributo
                melhor_threshold = threshold_atual
                melhor_X_esq = X_esq
                melhor_y_esq = y_esq
                melhor_X_dir = X_dir
                melhor_y_dir = y_dir
    
    return melhor_atributo, melhor_threshold, melhor_gini, melhor_X_esq, melhor_y_esq, melhor_X_dir, melhor_y_dir





def criterio_parada(criterio: int):
        if criterio == 1:
            print("mesma classe")
            return 1
        elif criterio == 2:
            print("profundidade max")
            return 2
        elif criterio == 3:
            print("melhoria de Gini")
            return 3

class ArvoreDecisaoClassificador:
    """
    Implementação de uma Árvore de Decisão para classificação usando o índice Gini.
    """
    
    def __init__(self, profundidade_max, min_amostras_split, min_gini_melhoria):
        """
        Construtor chucro

        Parameters
        ----------
        profundidade_max: int
            Profundidade máxima da árvore
        min_amostras_split: int
            Número mínimo de amostras necessárias para fazer um split
        min_gini_melhoria: float
            Melhoria mínima do Gini necessária para fazer um split
        """
        self.profundidade_max = profundidade_max
        self.min_amostras_split = min_amostras_split
        self.min_gini_melhoria = min_gini_melhoria
        self.raiz = None
    
    def fit(self, X, y):
        """
        Treina a árvore de decisão.
        
        Parameters
        ----------
        X: np.array
            Matriz de atributos (features)
        y: np.array
            Vetor de rótulos (classes)
        """
        X = np.array(X)
        y = np.array(y)
        self.raiz = self.construir_arvore(X, y, profundidade=0)
        return self
    
    def construir_arvore(self, X, y, profundidade):
        """
        Função recursiva para construção da árvore
        
        Parameters
        ----------
        X: np.array
            Matriz de atributos
        y: np.array
            Vetor de rótulos
        profundidade: int 
            Profundidade atual da árvore
        
        Returns 
        -------
        Node
            Nó da árvore (interno ou folha)
        """
        num_amostras = len(y)
        num_classes = len(np.unique(y))
        
        # Critérios de parada
        # 1 - Todas as amostras são da mesma classe
        if num_classes == 1:
            return Node(valor = y[0])
        
        # 2 - Profundidade máxima atingida
        if profundidade >= self.profundidade_max:
            classe_mais_comum = self.classe_mais_comum(y)
            return Node(valor=classe_mais_comum)
        
        # 3 - Número mínimo de amostras para split
        if num_amostras < self.min_amostras_split:
            classe_mais_comum = self.classe_mais_comum(y)
            return Node(valor=classe_mais_comum)
        
        # Encontrar o melhor split
        melhor_atributo, melhor_threshold, melhor_gini, X_esq, y_esq, X_dir, y_dir = melhor_split(X, y)
        
        # 4 - Não existe split válido
        if melhor_atributo is None:
            classe_mais_comum = self.classe_mais_comum(y)
            return Node(valor=classe_mais_comum)
        
        # 5 - Melhoria do Gini não é suficiente
        gini_atual = calcular_gini(y)
        melhoria = gini_atual - melhor_gini
        if melhoria < self.min_gini_melhoria:
            classe_mais_comum = self.classe_mais_comum(y)
            return Node(valor=classe_mais_comum)
        
        # Construir subárvores recursivamente
        X_esq = np.array(X_esq)
        y_esq = np.array(y_esq)
        X_dir = np.array(X_dir)
        y_dir = np.array(y_dir)
        
        no_esquerdo = self.construir_arvore(X_esq, y_esq, profundidade + 1)
        no_direito = self.construir_arvore(X_dir, y_dir, profundidade + 1)
        
        return Node(feature_index=melhor_atributo, limiar=melhor_threshold,esquerda=no_esquerdo,direita=no_direito)
    
    def classe_mais_comum(self, y):
        """
        Retorna a classe mais frequente em y.
        
        Parameters
        ----------
        y: np.array
            Vetor de rótulos
            
        Returns
        -------
        classe_mais_comum
            A classe que aparece mais vezes
        """
        valores, contagens = np.unique(y, return_counts=True)
        indice_max = np.argmax(contagens)
        return valores[indice_max]
    
    def predict(self, X):
        """
        Faz predições para as amostras em X
        
        Parameters
        ----------
        X: np.array
            Matriz de atributos para predição
            
        Returns
        -------
        np.array
            Vetor com as predições
        """
        X = np.array(X)
        return np.array([self.predizer_amostra(amostra, self.raiz) for amostra in X])
    
    def predizer_amostra(self, amostra, no):
        """
        Prediz a classe de uma única amostra navegando pela árvore
        
        Parameters
        ----------
        amostra: np.array
            Uma linha de atributos
        no: Node
            Nó atual da árvore
            
        Returns
        -------
        classe
            Classe predita para a amostra
        """
        # Se chegou em uma folha, retorna o valor
        if no.folha():
            return no.valor
        
        # Decidir se vai para esquerda ou direita
        if amostra[no.feature_index] <= no.limiar:
            return self.predizer_amostra(amostra, no.esquerda)
        else:
            return self.predizer_amostra(amostra, no.direita)
    
    # LLM
    def print_tree(self, no=None, profundidade=0):
        """
        Imprime a estrutura da árvore de forma visual.
        
        Parameters
        ----------
        no: Node
            Nó atual (None usa a raiz)
        profundidade: int
            Profundidade atual para indentação
        """
        if no is None:
            no = self.raiz
        
        if no is None:
            print("Árvore vazia!")
            return
        
        indentacao = "  " * profundidade
        
        if no.folha():
            print(f"{indentacao}Folha: classe = {no.valor}")
        else:
            print(f"{indentacao}Nó: atributo[{no.feature_index}] <= {no.limiar:.2f}")
            print(f"{indentacao}Esquerda:")
            self.print_tree(no.esquerda, profundidade + 1)
            print(f"{indentacao}Direita:")
            self.print_tree(no.direita, profundidade + 1)


class Node:
    """
    Classe que representa um nó da árvore de decisão.
    
    Nó interno -> possui feature_index e limiar, ainda pode ser dividido 
    Nó folha -> resultado final, só possui valor (predição)
    """
    def __init__(self, feature_index=None, limiar=None, esquerda=None, direita=None, valor=None):
        """
        Parameters
        ----------
        feature_index: int
            Índice do atributo usado para fazer o split neste nó
        limiar: float
            Valor de threshold para dividir os dados
        esquerda: Node
            Subárvore da esquerda (valores <= limiar)
        direita: Node
            Subárvore da direita (valores > limiar)
        valor: 
            Valor de predição para nó folha (classe mais comum)
        """
        self.feature_index = feature_index
        self.limiar = limiar
        self.esquerda = esquerda 
        self.direita = direita
        self.valor = valor
    
    def folha(self):
        """Verifica se o nó é uma folha (não tem filhos)"""
        return self.valor is not None

    


# Testes e exemplos de uso
if __name__ == "__main__":
    print("="*60)
    print("TESTANDO IMPLEMENTAÇÃO DA ÁRVORE DE DECISÃO")
    print("="*60)
    
    # Teste 1: Dataset simples de classificação binária
    print("\n[TESTE 1] Dataset simples - Classificação Binária")
    print("-"*60)
    
    X_treino = np.array([
        [2.5, 3.0],
        [1.5, 2.0],
        [3.5, 4.0],
        [3.0, 3.5],
        [1.0, 1.5],
        [4.0, 4.5],
        [2.0, 2.5],
        [3.8, 4.2],
        [1.2, 1.8],
        [3.3, 3.9]
    ])
    
    y_treino = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    
    print(f"Tamanho do dataset de treino: {len(X_treino)} amostras")
    print(f"Classes: {np.unique(y_treino)}")
    
    # Criar e treinar a árvore
    arvore = ArvoreDecisaoClassificador(
        profundidade_max=3, 
        min_amostras_split=2,
        min_gini_melhoria=0.0
    )
    
    print("\nTreinando a árvore...")
    arvore.fit(X_treino, y_treino)
    print("✓ Árvore treinada com sucesso!")
    
    # Visualizar a estrutura da árvore
    print("\nEstrutura da Árvore:")
    print("-"*60)
    arvore.print_tree()
    
    # Fazer predições no conjunto de treino (verificar overfitting)
    print("\nPredições no conjunto de treino:")
    print("-"*60)
    predicoes_treino = arvore.predict(X_treino)
    acuracia_treino = np.mean(predicoes_treino == y_treino)
    print(f"Acurácia no treino: {acuracia_treino*100:.2f}%")
    
    # Fazer predições em novos dados
    print("\nPredições em novos dados:")
    print("-"*60)
    X_teste = np.array([
        [1.8, 2.2],
        [3.2, 3.8],
        [1.2, 1.8],
        [4.1, 4.3]
    ])
    
    predicoes = arvore.predict(X_teste)
    for i, pred in enumerate(predicoes):
        print(f"Amostra {i+1}: {X_teste[i]} -> Classe predita: {pred}")
    
    # Teste 2: Testar funções auxiliares
    print("\n" + "="*60)
    print("[TESTE 2] Funções Auxiliares")
    print("-"*60)
    
    # Testar calcular_gini
    y_test_gini = np.array([0, 0, 0, 1, 1])
    gini = calcular_gini(y_test_gini)
    print(f"Gini de [0,0,0,1,1]: {gini:.4f}")
    print(f"  Esperado: ~0.48 (60% classe 0, 40% classe 1)")
    
    y_puro = np.array([1, 1, 1, 1])
    gini_puro = calcular_gini(y_puro)
    print(f"\nGini de [1,1,1,1]: {gini_puro:.4f}")
    print(f"  Esperado: 0.0 (conjunto puro)")
    
    # Testar dividir_dataset
    print("\n" + "-"*60)
    X_test_split = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_test_split = np.array([0, 0, 1, 1])
    X_esq, y_esq, X_dir, y_dir = dividir_dataset(X_test_split, y_test_split, 0, 4)
    print(f"Split no atributo 0, threshold=4:")
    print(f"  Esquerda: {len(y_esq)} amostras com classes {y_esq}")
    print(f"  Direita: {len(y_dir)} amostras com classes {y_dir}")
    
    # Teste 3: Dataset com 3 classes
    print("\n" + "="*60)
    print("[TESTE 3] Dataset com 3 Classes")
    print("-"*60)
    
    X_multiclass = np.array([
        [1, 1], [1.5, 1.5], [2, 2],  # Classe 0
        [5, 5], [5.5, 5.5], [6, 6],  # Classe 1
        [9, 9], [9.5, 9.5], [10, 10] # Classe 2
    ])
    
    y_multiclass = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    print(f"Dataset: {len(X_multiclass)} amostras, {len(np.unique(y_multiclass))} classes")
    
    arvore_multi = ArvoreDecisaoClassificador(
        profundidade_max=4,
        min_amostras_split=2,
        min_gini_melhoria=0.0
    )
    
    arvore_multi.fit(X_multiclass, y_multiclass)
    print("✓ Árvore multi-classe treinada!")
    
    predicoes_multi = arvore_multi.predict(X_multiclass)
    acuracia_multi = np.mean(predicoes_multi == y_multiclass)
    print(f"Acurácia: {acuracia_multi*100:.2f}%")
    
    print("\nEstrutura da árvore multi-classe:")
    print("-"*60)
    arvore_multi.print_tree()
    
    print("\n" + "="*60)
    print("TODOS OS TESTES CONCLUÍDOS COM SUCESSO! ✓")
    print("="*60)
