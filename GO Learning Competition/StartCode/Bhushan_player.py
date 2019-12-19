import sys
import random
import numpy as np
import itertools
class MyPlayer():
    move_history = None
    move_counter = None
    board_dictionary = None
    learning_rate = 0.2
    discount_factor = 0.8
    exploration_rate = [0.7, 0.3, 0.3, 0.2, 0.0]
    exploration_rate_counter = 1
    game_counter = 0
    def __init__(self):
        self.type = 'my'
        self.move_counter = 0
        self.load_dictionary_from_file()
        self.move_history = {}
        np.random.seed(1)
    def get_input(self, go, piece_type):
        '''
        Get one input.
        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''
        placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    placements.append((i,j))
        if len(placements)==0:
            print('this')
        move = None
        if np.random.uniform(0, 1) <= self.exploration_rate[self.exploration_rate_counter]:
            # randomly select one and apply it
            move = random.choice(placements)
        else:
            best_candidate_utility = -999
            for i,j in placements:
                test_go = go.copy_board()
                test_go.place_chess(i, j, piece_type)
                board_encoding = tuple(list(itertools.chain.from_iterable(test_go.board)))
                if board_encoding in self.board_dictionary:
                    candidate_utility = self.board_dictionary[board_encoding]
                else:
                    candidate_utility = 0
                if candidate_utility > best_candidate_utility:
                    best_candidate_utility = candidate_utility
                    move = (i,j)
        if move == None:
            print('that')
        test_go = go.copy_board()
        test_go.place_chess(move[0], move[1], piece_type)
        # test_go.remove_died_pieces(piece_type)
        self.move_history[self.move_counter] = tuple(list(itertools.chain.from_iterable(test_go.board)))
        self.move_counter += 1

        return move

    def learn(self, my_go,result):

        reward = self.assign_reward(result)
        # Get winning move
        move = self.move_history[self.move_counter - 1]
        if not move in self.board_dictionary:
            self.board_dictionary[move] = 0.0
        self.board_dictionary[move] = reward
        for x in range(self.move_counter - 2, -1, -1):

            move = self.move_history[x]
            if not move in self.board_dictionary:
                self.board_dictionary[move] = 0.0
            old_value_part1 = (1 - self.learning_rate) * self.board_dictionary[move]
            # old_value_part2 = (self.learning_rate * (reward + self.discount_factor * self.board_dictionary[move]))
            old_value_part2 = (self.learning_rate * (reward + self.discount_factor * self.board_dictionary[move]))
            self.board_dictionary[move] = old_value_part1 + old_value_part2
            # prev_reward = reward
            reward = self.board_dictionary[move]
        # Reset Move History
        self.move_history = {}
        self.move_counter = 0
        self.game_counter += 1
        if self.game_counter == 99:
            print(len(self.board_dictionary))
            self.save_dictionary_to_file()
            self.game_counter = 0
    # Flipped = true
    def assign_reward(self, result):
        if result != 1:
            return 1
        else:
            return -1
    def save_dictionary_to_file(self):
        # Save
        np.save('q_value_dict_qlearner_y_fixed_new.npy', self.board_dictionary)
    def load_dictionary_from_file(self):
        # Load
        try:
            self.board_dictionary = np.load('qValueFileForXSample.npy', allow_pickle='TRUE').item()
            # self.board_dictionary = np.load('q_value_dict.npy',
            #                                 allow_pickle='TRUE').item()
            print(len(self.board_dictionary.keys()))
        except (OSError, IOError) as e:
            self.board_dictionary = {}
# time python3 ./go_play.py -n 5 -p1 my -p2 random -t 50