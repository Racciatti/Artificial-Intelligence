from collections import deque

def bfs_generica(problema):    
    # Fila com (estado, caminho)
    fila = deque([(problema.getStartState(), [])])
    visitados = set()
    
    while fila:
        estado_atual, caminho = fila.popleft()
        
        # Já visitou? Pula
        if estado_atual in visitados:
            continue
        
        # Marca como visitado
        visitados.add(estado_atual)
        
        # É o objetivo? Retorna o caminho
        if problema.isGoalState(estado_atual):
            return caminho
        
        # Expande para próximos estados
        for proximo_estado, acao, custo in problema.expand(estado_atual):
            if proximo_estado not in visitados:
                novo_caminho = caminho + [acao]
                fila.append((proximo_estado, novo_caminho))
    
    return []  # Não achou solução


class ProblemaComGrafo:
    def __init__(self, grafo, inicio, fim):
        self.grafo = grafo
        self.inicio = inicio
        self.fim = fim
    
    def getStartState(self):
        return self.inicio
    
    def isGoalState(self, estado):
        return estado == self.fim
    
    def expand(self, estado):
        # Olha no grafo quais são os vizinhos
        vizinhos = []
        if estado in self.grafo:
            for vizinho in self.grafo[estado]:
                vizinhos.append((vizinho, f"ir_para_{vizinho}", 1))
        return vizinhos


# grafo = {
#     'Tarefa A': ['Tarefa B', 'Tarefa C'],
#     'Tarefa B': ['Tarefa A', 'Tarefa D', 'Tarefa E'],
#     'Tarefa C': ['Tarefa A', 'Tarefa F'],
#     'Tarefa D': ['Tarefa B'],
#     'Tarefa E': ['Tarefa B', 'Tarefa F'],
#     'Tarefa F': ['Tarefa C', 'Tarefa E']
# }

# problema_grafo = ProblemaComGrafo(grafo, 'Tarefa F', 'Tarefa B')
# solucao = bfs_generica(problema_grafo)
# print(f"Caminho no grafo: {solucao}")

