from Board import Board
import numpy as np

class QLearner:
    """  Your task is to implement `move()` and `learn()` methods, and 
         determine the number of games `GAME_NUM` needed to train the qlearner
         You can add whatever helper methods within this class.
    """


    # ======================================================================
    # ** set the number of games you want to train for your qlearner here **
    # ======================================================================
    GAME_NUM = 100
    OPPOSITE_SIDE = -1
    MY_SIDE = 1
    OPTIMAL_POSITION = 2


    def __init__(self):
        """ Do whatever you like here. e.g. initialize learning rate
        """
        # =========================================================
        self.gamma = 0.9
        self.learningRate = 0.2
        # =========================================================


    def buildInitialGrid(self, whichSide, board):
        initialGrid = [[0 for i in range(3)] for j in range(3)]
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] != whichSide:
                    initialGrid[i][j] = -1
                if board[i][j] == whichSide:
                    initialGrid[i][j] = 1

    def move(self, board):
        """ given the board, make the 'best' move 
            currently, qlearner behaves just like a random player  
            see `play()` method in TicTacToe.py 
        Parameters: board 
        """
        if board.game_over():
            return

        # =========================================================
        # ** Replace Your code here  **
        print(board)
        whichSide = self.side
        initialGrid = self.buildInitialGrid(whichSide, board)

        # 4. Instead of choosing moves randomly, use your q values to choose the best move
        # find all legal moves
        candidates = []
        for i in range(0, 3):
            for j in range(0, 3):
                if board.is_valid_move(i, j):
                    candidates.append(tuple([i, j]))

        # randomly select one and apply it
        idx = np.random.randint(len(candidates))
        move = candidates[idx]
        # =========================================================
        # 1. CODE to save move History


        return board.move(move[0], move[1], self.side)


    def learn(self, board):
        """ when the game ends, this method will be called to learn from the previous game i.e. update QValues
            see `play()` method in TicTacToe.py 
        Parameters: board
        """

        # 2. Traverse move history in reverse order and update q values
        # =========================================================
        #  
        # 
        # ** Your code here **
        #
        #
        # =========================================================

        # 3. Reset move history


    # do not change this function
    def set_side(self, side):
        self.side = side