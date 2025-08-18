import sys
import os
# Adicionar caminhos para todas as pastas
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'racciatti'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'milani'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'matheus'))

from problems import EightPuzzle
from agent import Agent
from solve import BFSStrategy, DFSStrategy

from UCS import ucs_generica,grafo_romania, ProblemaRomenia
from DLS_recursiva import depth_limited_search, ProblemaComGrafo
from IDS import iterative_deepening_search, grafo



def main():

    
    while True:
        print("\n----Menu de Busca no 8-puzzle----")
        print("1. Busca em largura (BFS)")
        print("2. Busca em Profundidade (DFS)")
        print("3. Busca de Custo Uniforme (UCS)")
        print("4. Busca de Aprofundamento Iterativo (IDS)")
        print("5. Busca em Profundidade Limitada (DLS)")
        print("0. Sair")

        op = input("Escolha uma opção: ")

        estrategia = None
        if op == '1':
            estrategia = BFSStrategy()
        elif op == '2':
            estrategia = DFSStrategy()
        elif op == '3':
            caminho, custo_total = ucs_generica(ProblemaRomenia(grafo_romania, 'Sibiu', 'Bucharest'))
            print(f"Caminho ótimo: {caminho}")
            print(f"Custo total: {custo_total}")
        elif op == '4':
            problema_grafo = ProblemaComGrafo(grafo, 'Tarefa A', 'Tarefa F')
            solucao = iterative_deepening_search(problema_grafo)
            print(f"Caminho encontrado usando IDS: {solucao}")
        elif op == '5':
            problema_grafo = ProblemaComGrafo(grafo, 'Tarefa A', 'Tarefa F')
            solucao = depth_limited_search(problema_grafo, 3)
            print(f"Caminho encontrado usando DLS: {solucao}")
        elif op == '0':
            print("saindo....")
            break
        else:
            print("Opção inválida. Tente novamente.")
            continue

        if estrategia:
            # Criar um problema do 8-puzzle
            problema = EightPuzzle(verbose=True)
            print(f"\nEstado inicial:")
            print(problema)
            
            # Criar agente e resolver
            agente = Agent()
            solution_path = agente.solve(problema, estrategia)
            
            if solution_path:
                print("\nSolução encontrada!")
                for i, state in enumerate(solution_path):
                    print(f"Passo {i}:\n{state}\n")
                print(f"Número de passos: {len(solution_path)-1}")
            else:
                print("Nenhuma solução encontrada.")

if __name__ == "__main__":
    main()
