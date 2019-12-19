# uncompyle6 version 3.5.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.3 (default, Apr  3 2019, 19:16:38) 
# [GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]
# Embedded file name: /Users/macoy/Desktop/Go/compile/smart_player.py
# Size of source mod 2**32: 4925 bytes


class SmartPlayer:

    def __init__(self):
        self.type = 'smart'

    def calculate_opponent_lose(self, go, piece_type, movements):
        test_go = go.copy_board()
        for i, j in movements:
            if not test_go.place_chess(i, j, piece_type):
                return -1

        return len(test_go.find_died_pieces(3 - piece_type))

    def valid_move_search(self, go, piece_type):
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    possible_placements.append((i, j))

        return possible_placements

    def minmax_move(self, go, piece_type):
        d2_possible_placements = self.valid_move_search(go, piece_type)
        max_score = 0
        max_score_moves = []
        for d2_possible_placement in d2_possible_placements:
            d2_test_go = go.copy_board()
            d2_test_go.place_chess(d2_possible_placement[0], d2_possible_placement[1], piece_type)
            d2_test_go.remove_died_pieces(3 - piece_type)
            d3_possible_placements = self.valid_move_search(d2_test_go, 3 - piece_type)
            min_score = go.size * go.size
            for d3_possible_placement in d3_possible_placements:
                d3_test_go = d2_test_go.copy_board()
                d3_test_go.place_chess(d3_possible_placement[0], d3_possible_placement[1], 3 - piece_type)
                d3_test_go.remove_died_pieces(piece_type)
                score = d3_test_go.score(piece_type)
                if score < min_score:
                    min_score = score

            if min_score == max_score:
                max_score_moves.append(d2_possible_placement)
            elif min_score > max_score:
                max_score_moves = [
                 d2_possible_placement]
                max_score = min_score

        return max_score_moves

    def greedy_search(self, go, piece_type):
        largest_died_chess_cnt = 0
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

        return (
         largest_died_chess_cnt, greedy_placements)

    def greedy_move(self, go, piece_type, possible_placements):
        depth1_greedy_kill_cnt, depth1_greedy_placements = self.greedy_search(go, piece_type)
        if depth1_greedy_kill_cnt > 0:
            return depth1_greedy_placements
        possible_placements = self.valid_move_search(go, piece_type)
        if len(possible_placements) == 1:
            return depth1_greedy_placements
        largest_kill_cnt = 0
        greedy_placements = []
        for movements in combinations(possible_placements, 2):
            kill_cnt = self.calculate_opponent_lose(go, piece_type, movements)
            if kill_cnt == largest_kill_cnt:
                greedy_placements.append(movements[0])
            elif kill_cnt > largest_kill_cnt:
                greedy_placements = [
                 movements[0]]
                largest_kill_cnt = kill_cnt

        if not greedy_placements:
            return possible_placements
        else:
            return greedy_placements

    def get_input(self, go, piece_type):
        """
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        """
        min_max_moves = self.minmax_move(go, piece_type)
        if len(min_max_moves) == 1:
            return min_max_moves[0]
        else:
            greedy_moves = self.greedy_move(go, piece_type, min_max_moves)
            return self.get_center_move(go.size, greedy_moves)

    def get_center_move(self, n, movements):
        center_moves = []
        shortest_distance_to_center = n * n
        for movement in movements:
            center_distance = abs(movement[0] - (n - 1) / 2) + abs(movement[1] - (n - 1) / 2)
            if center_distance == shortest_distance_to_center:
                center_moves.append(movement)
            if center_distance < shortest_distance_to_center:
                center_moves = [
                 movement]
                shortest_distance_to_center = center_distance

        return random.choice(center_moves)
# okay decompiling smart_player.pyc
