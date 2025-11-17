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
        if X[i][atributo] < threshold: 
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

    

def split_numerico():
    pass

def split_categorio():
    pass


# Exemplo de uso
if __name__ == "__main__":
    # Criar dados de exemplo simples
    # Vamos criar um dataset simples de classificação binária
    X_treino = np.array([
        [2.5, 3.0],
        [1.5, 2.0],
        [3.5, 4.0],
        [3.0, 3.5],
        [1.0, 1.5],
        [4.0, 4.5],
        [2.0, 2.5],
        [3.8, 4.2]
    ])
    
    y_treino = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    
    # Criar e treinar a árvore
    arvore = ArvoreDecisaoClassificador(profundidade_max=3, min_amostras_split=2)
    arvore.fit(X_treino, y_treino)
    
    # Visualizar a estrutura da árvore
    print("Estrutura da Árvore:")
    print("=" * 50)
    arvore.print_tree()
    
    # Fazer predições
    X_teste = np.array([
        [1.8, 2.2],
        [3.2, 3.8],
        [1.2, 1.8]
    ])
    
    predicoes = arvore.predict(X_teste)
    print("\n" + "=" * 50)
    print("Predições:")
    for i, pred in enumerate(predicoes):
        print(f"Amostra {i+1}: {X_teste[i]} -> Classe {pred}")

