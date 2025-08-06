import numpy as np
from abc import ABC, abstractmethod

class Node:
    """Um nó na árvore de busca."""

    def __init__(self, state, path, cost=0, depth=0):
        self.state = state
        self.path = path
        self.cost = cost
        self.depth = depth # Para DLS, IDS

class SearchStrategy(ABC):
    """Define a interface para todas as estratégias de busca."""

    @abstractmethod
    def add(self, node): pass
    
    @abstractmethod
    def remove(self): pass

    @abstractmethod
    def is_empty(self) -> bool: pass

class BFSStrategy(SearchStrategy):
    """Estratégia de Busca em Largura usando uma Fila."""

    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def remove(self):
        return self.frontier.pop(0)

    def is_empty(self) -> bool: return not self.frontier

class DFSStrategy(SearchStrategy):
    """Estratégia de Busca em Profundidade usando uma Pilha."""

    def __init__(self):
        self.frontier = []

    def add(self, node): self.frontier.append(node)
    def remove(self): return self.frontier.pop()
    def is_empty(self) -> bool: return not self.frontier
