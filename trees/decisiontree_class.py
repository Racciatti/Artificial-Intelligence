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
        if caracteristica not in proporcao:
            proporcao[caracteristica] = 0
        proporcao[caracteristica] += 1
    
    for classe in proporcao:
        proporcao[classe] /= total


    gini = 1 - sum(proporcao[classe]**2)

    return gini

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
    def __init__(self):
        pass

        


    def encontrar_melhor_split(X, y):
        pass

    def construir_arvore(x, y, *profundidade: int):
        """
        Função recursiva para construção da árvore
        Critério de parada: o nó é da mesma classe OU não existe um split melhor que o Gini
    
        Parameters
        ----------
        X: np.array
            Matriz de atributos, ou seja, features
        y: np.array
            Vetor de rótulos. Quantidade de classes.
        profundidae: int 
            Profundidade da árvore. Pode ser opcional, porque depende do critério de parada
        
        Returns 
        -------

        """

        # 1 - se o critério de parada for verdadeiro, retorna um nó folha
        # 2 - fazer divisão das classes 
        # 3 - Retornar recursivamente para o nó da direta ou esquerda



class Node:
    # Nó interno -> só não tem valor, pois ainda pode ser dividido 
    # Nó folha -> resultado final. Só possui valor
    def __init__(self, feature_index, limiar, esquerda, direita, valor):
        self.feature_index = feature_index # qual característica estamos analisando para o nó específico
        self.limiar = limiar
        self.esquerda = esquerda 
        self.direita = direita
        self.valor = valor

    

def split_numerico():
    pass

def split_categorio():
    pass
