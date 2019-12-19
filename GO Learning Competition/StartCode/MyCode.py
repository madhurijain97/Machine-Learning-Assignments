from itertools import combinations
import numpy

class MyPlayer():

    # setting the first and the second player as X and Y
    PLAYER_X, PLAYER_Y = 1, 2
    opponentPieceType = 0

    playerPlayed = 0
    boardDictionary, boardDictionaryForX, boardDictionaryForY = None, None, None
    fileNameForXPlayer, fileNameForYPlayer = 'qValueFileForXSample.npy', 'qValueFileForYSample.npy'

    # function used for copying the board and moving the chess piece
    def copyBoardAndPlayTheMove(self, go, move, piece_type):
        newBoard = go.copy_board()
        newBoard.place_chess(move[0], move[1], piece_type)
        return newBoard

    def __init__(self):
        self.type = 'my'
        self.boardDictionaryForX = numpy.load(self.fileNameForXPlayer, allow_pickle='TRUE').item()
        self.boardDictionaryForY = numpy.load(self.fileNameForYPlayer, allow_pickle='TRUE').item()
        numpy.random.seed(1)

    def removeDiedPiecesAfterPlaying(self, go, placements, pieceType, pieceTypeToBeRemoved):
        copiedBoard = self.copyBoardAndPlayTheMove(go, placements, pieceType)
        copiedBoard.remove_died_pieces(pieceTypeToBeRemoved)
        return copiedBoard


    def calculateLengthOfNumberOfDiedPieces(self, go, pieceType):
        return len(go.find_died_pieces(pieceType))

    def calculateOpponentsDiedPieces(self, go, piece_type, movements):
        copiedBoard = go.copy_board()
        for i, j in movements:
            if copiedBoard.place_chess(i, j, piece_type) == False:
                return -1
        return self.calculateLengthOfNumberOfDiedPieces(copiedBoard, self.opponentPieceType)

    # gives a list of valid moves by calling valid_place_check in go file
    def getValidCandidates(self, go, piece_type):
        validCandidates = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, True):
                    validCandidates.append((i, j))
        return validCandidates


    # playing opponent's move to predict the validity of our previous move
    def playOpponentsMove(self, opponentValidPlacements, copiedBoard, piece_type):
        minScore = float('inf')
        for opponentPlacement in opponentValidPlacements:
            copiedBoardOpponent = self.removeDiedPiecesAfterPlaying(copiedBoard, opponentPlacement, self.opponentPieceType, piece_type)
            score = copiedBoardOpponent.score(piece_type)
            if score < minScore:
                minScore = score
        return minScore

    def updateListsAndScores(self, finalValue, toBeComparedValue, toBeUpdatedlist, valueToBeAddedToList):
        if toBeComparedValue > finalValue:
            finalValue = toBeComparedValue
            toBeUpdatedlist = [valueToBeAddedToList]
        elif toBeComparedValue == finalValue:
            toBeUpdatedlist.append(valueToBeAddedToList)
        return finalValue, toBeUpdatedlist


    # running minimax algorithm
    def minimaxAlgo(self, go, piece_type):
        myPlayerValidPlacements = self.getValidCandidates(go, piece_type)
        maxScore = -float('inf')
        myPossibleGoodMoves = []
        for myPlacement in myPlayerValidPlacements:
            copiedBoard = self.removeDiedPiecesAfterPlaying(go, myPlacement, piece_type, self.opponentPieceType)
            minScore = self.playOpponentsMove(self.getValidCandidates(copiedBoard, self.opponentPieceType), copiedBoard, self.opponentPieceType)
            maxScore, myPossibleGoodMoves = self.updateListsAndScores(maxScore, minScore, myPossibleGoodMoves, myPlacement)

        return myPossibleGoodMoves

    def greedySearchAlgo(self, go, piece_type):
        largestDiedChessCount = 0
        greedyPlacements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, True):
                    copiedBoard = self.copyBoardAndPlayTheMove(go, (i, j), piece_type)
                    largestDiedChessCount, greedyPlacements = self.updateListsAndScores(largestDiedChessCount,
                                                                                        self.calculateLengthOfNumberOfDiedPieces(copiedBoard, self.opponentPieceType), greedyPlacements, (i, j))
        return largestDiedChessCount, greedyPlacements

    def performGreedyMove(self, go, piece_type):
        maximumNumberOfPiecesKilled, greedyPlacements = 0, []
        numberOfPiecesKilled, greedyPlacementMoves = self.greedySearchAlgo(go, piece_type)
        validPlacements = self.getValidCandidates(go, piece_type)

        if numberOfPiecesKilled > 0 or len(validPlacements) == 1:
            return greedyPlacementMoves

        # being greedy on two levels
        for movements in combinations(validPlacements, 2):
            piecesKilled = self.calculateOpponentsDiedPieces(go, piece_type, movements)
            maximumNumberOfPiecesKilled, greedyPlacements = self.updateListsAndScores(maximumNumberOfPiecesKilled, piecesKilled, greedyPlacements, movements[0])
        if greedyPlacements:
            return greedyPlacements
        return validPlacements


    # helps to choose a move randomly from a list of valid moves
    def chooseRandomly(self, candidates):
        idx = numpy.random.randint(len(candidates))
        chosenMove = candidates[idx]
        return chosenMove

    # to calculate distance such that we get a point with the minimum distance
    def calculateDistance(self, i, j, center):
        return abs(i - center) + abs(j - center)

    def findMinimumDistanceFromCentreOfTheBoard(self, boardSize, moves):
        centerMovesList = []
        shortestDistance, centerIndex = float('inf'), (boardSize - 1) / 2
        for move in moves:
            centerDistance = self.calculateDistance(move[0], move[1], centerIndex)
            if centerDistance < shortestDistance:
                centerMovesList = [move]
                shortestDistance = centerDistance
            elif centerDistance == shortestDistance:
                centerMovesList.append(move)
        return self.chooseRandomly(centerMovesList)

    # decide to play as minimax or greedy
    def playUsingMinimaxAndGreedy(self, go, piece_type):
        minimaxMoves = self.minimaxAlgo(go, piece_type)
        return minimaxMoves[0] if len(minimaxMoves) == 1 else self.findMinimumDistanceFromCentreOfTheBoard(go.size, self.performGreedyMove(go, piece_type))

    def encodeTheGoBoard(self, copiedBoard):
        encodedList = []
        for i in range(copiedBoard.size):
                encodedList.extend(copiedBoard.board[i])
        # return tuple(list(itertools.chain.from_iterable(copiedBoard.board)))
        return tuple(encodedList)

    # tic tac toe q learning used for go learning
    def playUsingQLearning(self, go, piece_type):
        placements = self.getValidCandidates(go, piece_type)
        maxQValue, bestMove = -float('inf'), None

        for validMove in placements:
            copiedBoard = self.copyBoardAndPlayTheMove(go, validMove, piece_type)
            board_encoding = self.encodeTheGoBoard(copiedBoard)
            if board_encoding not in self.boardDictionary:
                qValue = 0
            else:
                qValue = self.boardDictionary[board_encoding]
            if qValue > maxQValue:
                maxQValue = qValue
                bestMove = validMove

        return bestMove

        # load the player dictionary depending on the side on which we are playing
    def loadPlayerDictionary(self, player):
        if player == self.PLAYER_X:
            return self.boardDictionaryForX
        return self.boardDictionaryForY

    def decideIfQLearningToBeCalled(self, encodedBoardState):
        if encodedBoardState in self.boardDictionary:
            return True
        return False

    def get_input(self, go, piece_type):
        """
        Get one input.
        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        """

        self.opponentPieceType = 3 - piece_type
        validPlacements = self.getValidCandidates(go, piece_type)
        self.boardDictionary = self.loadPlayerDictionary(piece_type)
        for move in validPlacements:
            copiedBoard = self.copyBoardAndPlayTheMove(go, move, piece_type)
            encodedBoard = self.encodeTheGoBoard(copiedBoard)
            if self.decideIfQLearningToBeCalled(encodedBoard):
                return self.playUsingQLearning(go, piece_type)
        return self.playUsingMinimaxAndGreedy(go, piece_type)

