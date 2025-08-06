from problems import EightPuzzle
from solve import *

class Agent:
    """
    O agente que resolve um problema usando diferentes estratégias de busca.
    """

    def solve(self, problem: EightPuzzle, strategy: SearchStrategy):
        """Método de busca genérico que usa a estratégia fornecida."""

        start_state = Node(state=problem.state, path=[problem.state], cost=0, depth=0)

        strategy.add(start_state)
        
        visited = {tuple(problem.state.flatten())}

        while not strategy.is_empty():
            current_node = strategy.remove()
            
            if problem.isSolution(current_node.state):
                return current_node.path

            for next_state in problem.possibleMoves(current_node.state):
                state_tuple = tuple(next_state.flatten())
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    
                    new_cost = current_node.cost + 1
                    
                    new_node = Node(
                        state=next_state,
                        path=current_node.path + [next_state],
                        cost=new_cost,
                        depth=current_node.depth + 1
                    )
                    strategy.add(new_node)
        
        return None

    def solve_bfs(self, problem: EightPuzzle):
        return self._solve(problem, BFSStrategy())

    def solve_dfs(self, problem: EightPuzzle):
        return self._solve(problem, DFSStrategy())