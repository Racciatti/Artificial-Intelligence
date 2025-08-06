import numpy as np
import random

class EightPuzzle:

    def  __init__(self, initialState : np.ndarray = None, verbose : bool = False):

        if initialState is not None:
            if not self.isSolvable(initialState):

                raise ValueError("O estado inicial fornecido não é solucionável.")
            
            self.state = initialState

        else:
            self.state = self.generateRandomState()
    
        self.goal_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0]).reshape(3, 3)

        if verbose: print(self.__str__())

    def isSolvable(self, state : np.ndarray):
        """
        Checa a paridade do problema para definir se o estado atual é solucionável
        """
        flatState = state.flatten()

        numbers = flatState[flatState != 0]

        inversions = 0
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if numbers[i] > numbers[j]:
                    inversions += 1

        return inversions % 2 == 0

    def generateRandomState(self):
        """
        Gera um estado inicial aleatório para o problema, garantindo que ele seja solucionável
        """
        valid = False

        while not valid:

            # Generating random state
            numbers = [i for i in range(9)]
            random.shuffle(numbers)

            generatedState = np.array(numbers).reshape((3,3))
            if self.isSolvable(generatedState): valid = True

        return generatedState
        
    def isSolution(self, state) -> bool:
        return np.array_equal(state, self.goal_state)
    
    def possibleMoves(self, state):
        """
        Retorna quais são as ações válidas a partir do estado atual
        """
        possibleMoves = []
        
        # Find empty spot
        pos = np.where(state == 0)
        emptySpotRow, emptySpotCol = pos[0][0], pos[1][0]
        
        # Define possible moves
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # For each move
        for dRow, dCol in directions:
            newRow, newCol = emptySpotRow + dRow, emptySpotCol + dCol
            
            # Check if it is inside the board
            if 0 <= newRow < 3 and 0 <= newCol < 3:

                newState = state.copy()
                
                newState[emptySpotRow, emptySpotCol] = state[newRow, newCol]
                newState[newRow, newCol] = 0
                
                possibleMoves.append(newState)

        # Return all possible moves
        return possibleMoves

    def __str__(self):
        return f"current state: {str(self.state).replace('0',' ')}"


