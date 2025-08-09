import DLS_recursiva

def iterative_deepening_search(problem):
    profundidade = 0

    for profundidade in range(100): # colocar um limite máximo ao invés de infinito 
        resultado = DLS_recursiva.depth_limited_search(problem, profundidade)

        if resultado is not None:
            return resultado