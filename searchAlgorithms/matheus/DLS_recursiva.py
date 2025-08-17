def depth_limited_search(problem, profundidade_maxima):
    return recursive_dls(problem.initial_state(), problem, profundidade_maxima)

def recursive_dls(node, problem, profundidade_maxima):
    """
    Realiza a busca em profundidade usando recursividade

    -----
    Params:
        node: O nó atual a ser explorado
        problem: Uma instância do problema que define o espaço de busca, incluindo estado inicial, ações disponíveis, função de transição e teste de objetivo
        profundidade_maxima: O limite de profundidade para a busca
    
    Returns:
        None: não foi encontrado nenhum caminho
        node['PATH']: retorna o caminho do nó objetivo
    """
    if problem.is_goal_state(node['STATE']):
        return node['PATH']

    if profundidade_maxima == 0:
        return None # cutoff - limite da profundidade foi alcançada

    for acao in problem.actions(node['STATE']):
        node_filho = problem.result(node['STATE'], acao)
        if node_filho is not None:
            # Atualiza o caminho do nó filho
            node_filho['PATH'] = node['PATH'] + [acao]
            resultado = recursive_dls(node_filho, problem, profundidade_maxima - 1)
            if resultado is not None:
                return resultado

    return None # nenhum resultado encontrado

class ProblemaComGrafo:
    # construtor
    def __init__(self, grafo, inicio, fim):
        self.grafo = grafo 
        self.inicio = inicio 
        self.fim = fim

    # Retorna o estado inicial do problema
    def initial_state(self):
        return {'STATE': self.inicio, 'PATH': []} 

    # Retorna o estado final do problema
    def is_goal_state(self, estado):
        return estado == self.fim
    
    # função ACTIONS (do slide)
    def actions(self, estado):
        # Retorna as ações disponíveis a partir do estado atual
        return self.grafo.get(estado, [])

    # função RESULT
    def result(self, estado, acao):
        if estado in self.grafo and acao in self.grafo[estado]: # Verifica se o estado e a ação são válidos
            # se a ação é válida, retorna o novo nó com estado e caminho atualizados
            novo_estado = acao  # a ação é o próprio estado de destino
            return {'STATE': novo_estado, 'PATH': []} # PATH será construído na função recursive_dls
        return None


## PROBLEMA IGUAL O DO MILANI PARA TESTE    
grafo = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

problema_grafo = ProblemaComGrafo(grafo, 'A', 'F')
solucao = depth_limited_search(problema_grafo, 3)
print(f"Caminho usando DLS (recursiva): {solucao}")