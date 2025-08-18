import heapq

def ucs_generica(problema):    
    # Fila de prioridade com (custo_acumulado, estado, caminho)
    # heapq usa o primeiro elemento (custo) para ordenação
    fila_prioridade = [(0, problema.getStartState(), [])]
    visitados = set()
    
    while fila_prioridade:
        custo_atual, estado_atual, caminho = heapq.heappop(fila_prioridade)
        
        # Já visitou? Pula
        if estado_atual in visitados:
            continue
        
        # Marca como visitado
        visitados.add(estado_atual)
        
        # É o objetivo? Retorna o caminho
        if problema.isGoalState(estado_atual):
            return caminho, custo_atual
        
        # Expande para próximos estados
        for proximo_estado, acao, custo_passo in problema.expand(estado_atual):
            if proximo_estado not in visitados:
                novo_custo = custo_atual + custo_passo
                novo_caminho = caminho + [acao]
                heapq.heappush(fila_prioridade, (novo_custo, proximo_estado, novo_caminho))
    
    return [], float('inf')  # Não achou solução


class ProblemaRomenia:
    def __init__(self, grafo, inicio, fim):
        self.grafo = grafo
        self.inicio = inicio
        self.fim = fim
    
    def getStartState(self):
        return self.inicio
    
    def isGoalState(self, estado):
        return estado == self.fim
    
    def expand(self, estado):
        # Retorna (vizinho, acao, custo)
        vizinhos = []
        if estado in self.grafo:
            for vizinho, custo in self.grafo[estado]:
                vizinhos.append((vizinho, f"{estado}→{vizinho}", custo))
        return vizinhos


grafo_romania = {
    'Sibiu': [('Rimnicu Vilcea', 80), ('Fagaras', 99)],
    'Rimnicu Vilcea': [('Sibiu', 80), ('Pitesti', 97)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Pitesti': [('Rimnicu Vilcea', 97), ('Bucharest', 101)],
    'Bucharest': [('Fagaras', 211), ('Pitesti', 101)]
}

# problema_romania = ProblemaRomenia(grafo_romania, 'Sibiu', 'Bucharest')
# caminho, custo_total = ucs_generica(problema_romania)
# print(f"Caminho ótimo: {caminho}")
# print(f"Custo total: {custo_total}")
