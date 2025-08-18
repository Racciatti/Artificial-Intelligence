from DLS_recursiva import depth_limited_search as dls_r, ProblemaComGrafo

def iterative_deepening_search(problem):
    profundidade = 0

    for profundidade in range(100): # colocar um limite máximo ao invés de infinito 
        resultado = dls_r(problem, profundidade)

        if resultado is not None:
            return resultado

grafo = {
    'Tarefa A': ['Tarefa B', 'Tarefa C'],
    'Tarefa B': ['Tarefa A', 'Tarefa D', 'Tarefa E'],
    'Tarefa C': ['Tarefa A', 'Tarefa F'],
    'Tarefa D': ['Tarefa B'],
    'Tarefa E': ['Tarefa B', 'Tarefa F'],
    'Tarefa F': ['Tarefa C', 'Tarefa E']
}

# problema_grafo = ProblemaComGrafo(grafo, 'Tarefa A', 'Tarefa F')
# solucao = iterative_deepening_search(problema_grafo)
# print(f"Caminho encontrado usando ILS: {solucao}")