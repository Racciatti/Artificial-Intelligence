from DLS_recursiva import depth_limited_search as dls_r, ProblemaComGrafo

def iterative_deepening_search(problem):
    profundidade = 0

    for profundidade in range(100): # colocar um limite máximo ao invés de infinito 
        resultado = dls_r(problem, profundidade)

        if resultado is not None:
            return resultado

grafo = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

problema_grafo = ProblemaComGrafo(grafo, 'A', 'F')
solucao = iterative_deepening_search(problema_grafo)
print(f"Caminho encontrado usando ILS: {solucao}")