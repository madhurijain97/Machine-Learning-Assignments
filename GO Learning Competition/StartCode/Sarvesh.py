# INF - 552 | Machine Learning
# Competition | Fall 2019
# @author - Sarvesh Parab (sparab@usc.edu)

import sys
import pickle
import os
import numpy as np
import random


class MyPlayer():
    def __init__(self):
        # Set player type
        self.type = 'my'

        # Initialize Numpy Seed
        np.random.seed(1)

        # Init model parameters
        self.ALPHA = 0.2
        self.GAMMA = 0.8
        self.EPSILON = 0.0
        self.INIT_Q = 0.0
        self.DUMP_TO_PKL_FREQ = 1000

        # Board parameters
        self.BOARD_SIZE = 5

        # Set reward values
        self.WIN_REWARD = 20.0
        self.DRAW_REWARD = 0.0
        self.LOSS_REWARD = -40.0

        # Game state Tracking
        self.GAME_STATE_ONGOING = -1
        self.GAME_STATE_DRAW = 0
        self.GAME_STATE_BLACK_WIN = 1
        self.GAME_STATE_WHITE_WIN = 2

        # Player Identifiers
        self.PLAYER_BLACK = 1
        self.PLAYER_WHITE = 2

        # Single variable to keep track of just the previous state to the game over state
        self.past_state = list()

        # Single variable to keep track of just the previous action to the game over action
        self.past_action = -1

        # List to keep track of different states the game for this player goes through till game over
        self.state_history = list()

        # List to keep track of different actions taken by this player in the game till game over
        self.action_history = list()

        # A dictionary data structure to map board encoded state for both sides, (state, action) v/s q_value
        self.q_table = dict()

        # Location of the q table pickle dump
        self.q_table_pkl = 'qTable.pkl'

        # A simple counter as to number of times 'my player' has played a game
        self.games_played = 0

        # Load to the q-table
        self.data_dict = self.load_q_table()
        self.q_table = self.data_dict['q_table']
        self.games_played = self.data_dict['games_played']

    #
    def get_input(self, go, piece_type):
        # Init the past state with the current board state
        self.past_state = np.array([cell for sublist in go.board for cell in sublist])

        # Check if any learnt knowledge is successfully loaded or not
        if len(self.q_table) == 0:
            # print('Loaded Q-Table has no stored knowledge | Reverting to random moves')

            possible_placements = []
            for i in range(go.size):
                for j in range(go.size):
                    if go.valid_place_check(i, j, piece_type, test_check=True):
                        possible_placements.append((i, j))

            move = random.choice(possible_placements)

            self.past_action = move[0] * self.BOARD_SIZE + move[1]

        #
        else:
            # Find all unassigned positions on the board
            empty_cells = np.where(self.past_state == 0)[0]

            # Randomly chose move if below the epsilon threshold
            rand_val = np.random.uniform()
            if self.games_played <= self.DUMP_TO_PKL_FREQ and rand_val < self.EPSILON:
                self.past_action = np.random.choice(empty_cells)
            else:
                # Fetch all q values for these empty cells for the current state
                q_vals = np.array([self.q_table.get((tuple(self.past_state), ec), self.INIT_Q) for ec in empty_cells])

                # Choose the indices with the highest q value
                highest_q_vals = np.argwhere(q_vals == np.amax(q_vals)).flatten()

                # If multiple q values are maximum, then randomly choose one, else pick the highest
                self.past_action = empty_cells[np.random.choice(highest_q_vals)]

            # Get the coordinates to play the action
            move = (int(self.past_action / self.BOARD_SIZE), int(self.past_action % self.BOARD_SIZE))

        # collect the game history
        self.state_history.append(self.past_state)
        self.action_history.append(self.past_action)

        return move

    #
    def learn(self, go, piece_type, result):

        # Game play counter
        self.games_played += 1

        # Fetch the reward value for the game turnout
        reward_val = self.get_reward_value(piece_type, result)

        # Handle separately for terminal node and update the respective q value
        terminal_state = tuple(self.state_history[-1])
        terminal_action = self.action_history[-1]

        terminal_q_val = self.q_table.get((terminal_state, terminal_action), self.INIT_Q)
        self.q_table[(terminal_state, terminal_action)] = reward_val

        # Remove the processed terminal history
        del self.state_history[-1]
        del self.action_history[-1]

        # Propagate reward backwards through move history and update the respective q values
        for s, a in zip(self.state_history[::-1], self.action_history[::-1]):
            intermediate_q_val = self.q_table.get((tuple(s), a), self.INIT_Q)

            # Get the max q value for the state reached on applying the action 'a' on state 's' for all possible actions
            new_state = np.array(s, copy=True)
            new_state[a] = piece_type
            next_state_q_vals = [self.q_table.get((tuple(new_state), poss_action), self.INIT_Q) for poss_action in
                                 np.where(new_state == 0)[0]]
            maximal_q_val_for_next_state = max(next_state_q_vals) if len(next_state_q_vals) > 0 else self.INIT_Q

            # Update the reward value
            reward_val = self.GAMMA * reward_val

            # Update the q value for the intermediate state
            self.q_table[(tuple(s), a)] = (1.0 - self.ALPHA) * intermediate_q_val + self.ALPHA * reward_val \
                                          + self.ALPHA * self.GAMMA * maximal_q_val_for_next_state

        # Clear the history data structures
        self.state_history.clear()
        self.action_history.clear()

        # Save generated q table as pickle
        if self.games_played % self.DUMP_TO_PKL_FREQ == 0:
            self.save_q_table()

    # Utility method to load saved knowledge in form of the pickled Q table
    def load_q_table(self):
        if not os.path.isfile(self.q_table_pkl):
            self.save_q_table()
        with open(self.q_table_pkl, 'rb') as fp:
            data_dict = pickle.load(fp)

        q_val_keys = data_dict['q_table'].keys()

        print("\t\tLoaded unique q values : " + str(len(q_val_keys)))
        print("\t\tloaded unique states : " + str(len(set([state[0] for state in q_val_keys]))))

        return data_dict

    # Utility method to dump the Q table into a pickled file
    def save_q_table(self):
        q_val_keys = self.q_table.keys()
        print("\tGames played : " + str(self.games_played) + '\n')
        print("\t\tSaved unique q values : " + str(len(q_val_keys)))
        print("\t\tSaved unique states : " + str(len(set([state[0] for state in q_val_keys]))))
        print()
        with open(self.q_table_pkl, 'wb') as fp:
            pickle.dump({'q_table': self.q_table,
                         'games_played': self.games_played}, fp)

    # Utility function to check who won and get the reward value
    def get_reward_value(self, piece_type, result):
        if (result == self.GAME_STATE_WHITE_WIN and piece_type == self.PLAYER_WHITE) \
                or (result == self.GAME_STATE_BLACK_WIN and piece_type == self.PLAYER_BLACK):
            reward_val = self.WIN_REWARD
        elif (result == self.GAME_STATE_WHITE_WIN and piece_type == self.PLAYER_BLACK) \
                or (result == self.GAME_STATE_BLACK_WIN and piece_type == self.PLAYER_WHITE):
            reward_val = self.LOSS_REWARD
        elif result == self.GAME_STATE_DRAW:
            reward_val = self.DRAW_REWARD

        return reward_val

    # Reset all to a state as if no game ever played
    def hard_reset_my_player(self):
        os.remove(self.q_table_pkl)
        self.games_played = 0