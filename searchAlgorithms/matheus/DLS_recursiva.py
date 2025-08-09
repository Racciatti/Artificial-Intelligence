def depth_limited_search(problem, profundidade_maxima):
    return recursive_dls(problem.initial_state(), problem, profundidade_maxima)

def recursive_dls(node, problem, profundidade_maxima):
    """
    Realiza a busca em profundidade usando recursão

    -----
    Params:
        node: O nó atual a ser explorado
        problem: O problema a ser resolvido
        profundidade_maxima: O limite de profundidade para a busca
    

    """
    if problem.is_goal_state(node['STATE']):
        return node['PATH']

    if profundidade_maxima == 0:
        return None # cutoff - limite da profundidade foi alcançada

    for acao in problem.actions(node['STATE']):
        node_filho = problem.result(node['STATE'], acao)
        resultado = recursive_dls(node_filho, problem, profundidade_maxima - 1)
        if resultado is not None:
            return resultado

    return None # nenhum resultado encontrado