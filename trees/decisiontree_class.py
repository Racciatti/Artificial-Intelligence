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

class ArvoreDecisaoClassificador:
    def __init__(self):
        pass

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
        
    def melhor_gini():
        return True

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
            Vetro de rótulos. Quantidade de classes.
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
