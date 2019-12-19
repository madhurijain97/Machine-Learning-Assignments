import random
import numpy as np
from collections import defaultdict

class QLearner:
    """ Your task is to implement `move()` and `learn()` methods, and
        determine the number of games `GAME_NUM` needed to train the qlearner
        You can add whatever helper methods within this class.
    """

    GAME_NUM = 10000
    PLAYER_X, X_WIN = 1, 1
    PLAYER_O, O_WIN = 2, 2
    moveHistoryList = []
    moveHistory = {}
    possibleActions = defaultdict(list)
    np.random.seed(1)

    def __init__(self):
        """ Do whatever you like here. e.g. initialize learning rate
        """
        self.learningRate = 0.4
        self.gamma = 0.8
        self.epsilon = 0.0

    # Find total valid moves and return them as a list
    def findLegalMoves(self, board):
        candidates = []
        for i in range(0, 3):
            for j in range(0, 3):
                if board.is_valid_move(i, j):
                    candidates.append(tuple([i, j]))
        return candidates

    # For agent to learn and explore, it will pick a random move from all the valid moves
    def chooseRandomly(self, candidates):
        idx = np.random.randint(len(candidates))
        chosenMove = candidates[idx]
        return chosenMove

    # Choose moves based on Q values that were already computed in the previous games
    # Choose moves with the highest q value in valid moves
    def chooseBasedOnQValue(self, candidates, board):
        maxQValue = -float('inf')
        action = None
        for validMove in candidates:
            if (board.encode_state(), validMove) not in self.moveHistory:
                qValue = 0
            else:
                qValue = self.moveHistory[(board.encode_state(), validMove)]

            if qValue > maxQValue:
                maxQValue = qValue
                action = validMove
        return action

    # Will decide based on exploration rate, if we have to choose the action randomly or based on Q learning
    def chooseAction(self, candidates, board):
        if (random.random() < self.epsilon):
            chosenMove = self.chooseRandomly(candidates)
        else:
            chosenMove = self.chooseBasedOnQValue(candidates, board)
        return chosenMove


    def move(self, board):
        """ given the board, make the 'best' move
            currently, qlearner behaves just like a random player
            see `play()` method in TicTacToe.py
            Parameters: board
        """
        if board.game_over():
            return

        candidates = self.findLegalMoves(board)
        chosenMove = self.chooseAction(candidates, board)
        self.moveHistoryList.append((board.encode_state(), chosenMove))
        self.possibleActions[(board.encode_state(), chosenMove)].append(candidates)
        return board.move(chosenMove[0], chosenMove[1], self.side)

    #after learning has been done for one game, reset all the lists and dictionaries that were used
    def resetLists(self):
        self.moveHistoryList = []
        self.possibleActions.clear()
        self.possibleActions = defaultdict(list)

    # Based on which side won and which side you belong to, assign the rewards
    def decideReward(self, board):
        gameResult = board.game_result
        if gameResult == self.X_WIN:
            if self.side == self.PLAYER_X:
                reward = 10
            else:
                reward = -10
        elif gameResult == self.O_WIN:
            if self.side == self.PLAYER_O:
                reward = 10
            else:
                reward = -10
        else:
            reward = 5
        return reward

    # Choose the move with the maximum Q value in the next state from the current state
    def findMaximumQValueOfNextState(self, st, state, index):
        maxQValue = -float('inf')
        for validMove in self.possibleActions[st][0]:
            if (state, validMove) not in self.moveHistory:
                qValue = 0
            else:
                qValue = self.moveHistory[(state, validMove)]
            if qValue > maxQValue:
                maxQValue = qValue
        return maxQValue

    #will iterate through the moveHistory and update their q values based on the q learning formula
    def performQLearning(self, reward):
        self.moveHistory[self.moveHistoryList[-1]] = reward
        for i, st in reversed(list(enumerate(self.moveHistoryList[:-1]))):
            if st not in self.moveHistory:
                self.moveHistory[st] = 0
            self.moveHistory[st] = (1 - self.learningRate) * self.moveHistory[st] + \
                                   self.learningRate * (reward + self.gamma * self.findMaximumQValueOfNextState(st, self.moveHistoryList[i+1], i+1))

    def learn(self, board):
        """ when the game ends, this method will be called to learn from the previous game i.e. update QValues
            see `play()` method in TicTacToe.py
            Parameters: board
        """
        reward = self.decideReward(board)
        self.performQLearning(reward)
        self.resetLists()

    # do not change this function
    def set_side(self, side):
        self.side = side