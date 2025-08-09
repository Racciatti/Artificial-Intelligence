# DFS - Depth First Search (busca em profundidade)

from multiprocessing import util

def depth_first_search(problem):
    """
    Realiza a busca em profundidade no problema fornecido.

    ------
    Params:
        problem: O problema a ser resolvido

    Returns:
        None: caso a função não encontre uma solução
    """
    node = get_start_node(problem)

    fronteira = util.Stack()
    fronteira.push(node)
    explorado = set() ## já visitou o nó

    # enquanto o nó de fronteira não for explorado por completo (entenda por estar vazio)
    while not fronteira.is_empty():
        # expande o nó da fronteira que está na fila
        node = fronteira.pop()

        # se nó já foi explorado (definido pelo conjunto explorado), vamos continuar
        if node['STATE'] in explorado: 
            continue
        
        explorado.add(node['STATE'])

        # o nó é o estado objetivo? Se for, vamos retornar o caminho até esse nó
        if problem.is_goal_state(node['STATE']):
            return node['PATH']
        
        for sucessor in problem.expand(node['STATE']):
            node_filho = get_child_node(sucessor, node)
            fronteira.push(node_filho)

    # se dentro do while não encontramos nada, retornamos vazio 
    return []

class ProblemaComGrafo:
    def __init__(self, grafo, estado_inicial, estado_objetivo):
        """
        Inicializa o problema com um grafo, estado inicial e estado objetivo.

        -------
        Params:
            grafo: O grafo que representa o problema
            estado_inicial: O estado inicial do problema
            estado_objetivo: O estado objetivo do problema

        Returns:
            None
        """
        self.grafo = grafo 
        self.estado_inicial = estado_inicial 
        self.estado_objetivo = estado_objetivo 

    # Retorna o estado inicial do problema
    def get_start_node(self):
        """
        Retorna o estado inicial do problema

        -------
        Returns:
            estado_inicial: O estado inicial do problema
        """
        return self.estado_inicial

    def is_goal_state(self, estado):
        """
        Verifica se o estado ATUAL é o estado objetivo do problema

        -------
        Params:
            estado: O estado a ser verificado

        Returns:
            bool: True se o estado for o objetivo, False caso contrário
        """
        return estado == self.estado_objetivo

    def expand(self, estado):
        """
        Faz a expansão do nó, fornecendo seus nós-filhos

        -------
        Params:
            estado: O nó a ser expandido

        Returns:
            list: Uma lista de nós-filhos resultantes da expansão
        """
        return self.grafo.get(estado, [])