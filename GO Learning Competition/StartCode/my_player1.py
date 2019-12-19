import random
import numpy as np
from collections import defaultdict
import pickle


class MyPlayer():

    GAME_NUM = 10000
    PLAYER_X, X_WIN = 1, 1
    PLAYER_O, O_WIN = 2, 2
    gameCount, saveInGameNumbers = 0, 500
    moveHistoryList = []
    moveHistory = {}
    fileName = 'qLearning.pkl'
    possibleActions = defaultdict(list)
    np.random.seed(1)

    def __init__(self):
        self.type = 'my'
        self.learningRate = 0.4
        self.gamma = 0.8
        self.epsilon = 0.0

    def greedy_search(self, go, piece_type):
        largest_died_chess_cnt = 0
        died_chess_cnt = 0
        greedy_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    test_go = go.copy_board()
                    test_go.place_chess(i, j, piece_type)
                    died_chess_cnt = len(test_go.find_died_pieces(3 - piece_type))
                if died_chess_cnt == largest_died_chess_cnt:
                    greedy_placements.append((i, j))
                elif died_chess_cnt > largest_died_chess_cnt:
                    largest_died_chess_cnt = died_chess_cnt
                    greedy_placements = [(i, j)]

        return greedy_placements

    # Find total valid moves and return them as a list
    def findLegalMoves(self, pieceType, go):
        candidates = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, pieceType, test_check=True):
                    candidates.append((i, j))
        return candidates

    # For agent to learn and explore, it will pick a random move from all the valid moves
    def chooseRandomly(self, candidates):
        idx = np.random.randint(len(candidates))
        chosenMove = candidates[idx]
        return chosenMove

    # Choose moves based on Q values that were already computed in the previous games
    # Choose moves with the highest q value in valid moves
    def chooseBasedOnQValue(self, candidates, go, pieceType):
        maxQValue = -float('inf')
        action = None
        for validMove in candidates:
            if (self.encode_state(go), validMove) not in self.moveHistory:
                qValue = 0
            else:
                qValue = self.moveHistory[(self.encode_state(go), validMove)]

            if qValue > maxQValue:
                maxQValue = qValue
                action = validMove

        if maxQValue == 0:
            greedyPlacements = self.greedy_search(go, pieceType)
            if len(greedyPlacements) == 1:
                return greedyPlacements[0]
            else:
                return self.chooseRandomly(greedyPlacements)
        return action

    # Will decide based on exploration rate, if we have to choose the action randomly or based on Q learning
    def chooseAction(self, candidates, go, pieceType):
        if (random.random() < self.epsilon):
            chosenMove = self.chooseRandomly(candidates)
        else:
            chosenMove = self.chooseBasedOnQValue(candidates, go, pieceType)
        return chosenMove

    def encode_state(self, go):
        """ Encode the current state of the board as a string
        """
        encodedState = ''
        for i in range(go.size):
            for j in range(go.size):
                encodedState = encodedState + (str(go.board[i][j]))
        return encodedState

    def get_input(self, go, pieceType):
        """ given the board, make the 'best' move
            currently, qlearner behaves just like a random player
            see `play()` method in TicTacToe.py
            Parameters: board
        """
        candidates = self.findLegalMoves(pieceType, go)
        dataDictionary = self.readFromPickleFile()
        if(len(dataDictionary['moveHistory'].keys()) != 0 and self.gameCount == 0):
            self.moveHistory = dataDictionary['moveHistory']
        chosenMove = self.chooseAction(candidates, go, pieceType)
        self.moveHistoryList.append((self.encode_state(go), chosenMove))
        self.possibleActions[(self.encode_state(go), chosenMove)].append(candidates)
        return chosenMove[0], chosenMove[1]

    def saveToPickle(self):
        with open(self.fileName, 'wb') as fp:
            pickle.dump({'moveHistory': self.moveHistory}, fp)

    # after learning has been done for one game, reset all the lists and dictionaries that were used
    def resetLists(self):
        self.moveHistoryList = []
        self.possibleActions.clear()
        self.possibleActions = defaultdict(list)

    # Based on which side won and which side you belong to, assign the rewards
    def decideReward(self, winner, pieceType):
        if winner == pieceType and (winner == 1 or winner == 2):
            reward = 10
        elif winner != pieceType and (winner == 1 or winner == 2):
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

    # will iterate through the moveHistory and update their q values based on the q learning formula
    def performQLearning(self, reward):
        self.moveHistory[self.moveHistoryList[-1]] = reward
        for i, st in reversed(list(enumerate(self.moveHistoryList[:-1]))):
            if st not in self.moveHistory:
                self.moveHistory[st] = 0
            self.moveHistory[st] = (1 - self.learningRate) * self.moveHistory[st] + \
                                   self.learningRate * (reward + self.gamma * self.findMaximumQValueOfNextState(st, self.moveHistoryList[i + 1], i + 1))

    def learn(self, result, pieceType):
        """ when the game ends, this method will be called to learn from the previous game i.e. update QValues
            see `play()` method in TicTacToe.py
            Parameters: board
        """
        reward = self.decideReward(result, pieceType)
        self.performQLearning(reward)
        self.resetLists()
        self.gameCount += 1
        if(self.gameCount == self.saveInGameNumbers):
            self.gameCount = 0
            self.saveToPickle()

    def readFromPickleFile(self):
        with open(self.fileName, 'rb') as fp:
            dataDictionary = pickle.load(fp)

        qLearningKeys = dataDictionary['moveHistory'].keys()
        return dataDictionary
        # print(qLearningKeys)


    # def get_input(self, go, piece_type):
    #     """
    #     Get one input.
    #
    #     :param go: Go instance.
    #     :param piece_type: 1('X') or 2('O').
    #     :return: (row, column) coordinate of input.
    #     """

