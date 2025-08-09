# DFS - Depth First Search (busca em profundidade)

from multiprocessing import util

def depth_first_search(problem):
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
            no_filho = get_child_node(sucessor, node)
            fronteira.push(no_filho)

    # se dentro do while não encontramos nada, retornamos vazio 
    return []