import numpy as np
import os
actions_count = 9

# DIFFERENT ACTIONS:
# ----------------------------------------
# | 0 = (0, 0) | 1 = (0, 1) | 2 = (0, 2) |
# ----------------------------------------
# | 3 = (1, 0) | 4 = (1, 1) | 5 = (1, 2) |
# ----------------------------------------
# | 6 = (2, 0) | 7 = (2, 1) | 8 = (2, 2) |
# ----------------------------------------
# In general, action a = ( a // 3, a % 3 )


class TicTacToeEnv:

    def __init__(self, width=3, height=3):
        self.width = width
        self.height = height
        self.board = np.full((self.width, self.height), 0)

    def get_current_player(self, state=None):
        state = state if state is not None else self.board.copy()

        # Unique will contain the values that are currently on
        # the board (most likely [0, 1, 2]), and counts will
        # contain how many times each vaule occurs on the board
        unique, counts = np.unique(state, return_counts=True)

        player1_count = counts[np.where(unique == 1)]
        player2_count = counts[np.where(unique == 2)]

        if len(player1_count) == 0:
            player1_count = 0
        if len(player2_count) == 0:
            player2_count = 0
        if player1_count > player2_count:
            return 2
        else:
            return 1

    def _to_idx(self, a):
        return (a // 3, a % 3)

    def step(self, a):
        step_player = self.get_current_player()

        # In case an invalid column index is provided
        if a not in range(actions_count):
            state = self.board.copy()
            reward = -1
            result = -1
            return state, reward, result

        # In case the given cell already has a pawn
        row_idx, col_idx = self._to_idx(a)
        if self.board[row_idx, col_idx] != 0:
            state = self.board.copy()
            reward = -1
            result = -1
            return state, reward, result

        self.board[row_idx, col_idx] = step_player
        state = self.board.copy()
        result = self._judge(row_idx, col_idx)
        if result == step_player:
            reward = 1
        else:
            reward = 0

        return state, reward, result

    # Determine if the player that just played is the winner
    def _judge(self, row_idx, col_idx):
        result = 0
        player = self.board[row_idx, col_idx]

        # Vertical:
        for col in range(3):
            if np.sum(self.board[:, col] == player) == 3:
                return player

        # Horizontal
        for row in range(3):
            if np.sum(self.board[row] == player) == 3:
                return player

        # Main Diagonal
        if np.sum(self.board.diagonal() == player) == 3:
            return player

        # Other Diagonal
        if np.sum(np.diag(np.fliplr(self.board)) == player) == 3:
            return player

        # Check for draw
        if np.min(self.board) > 0:
            return 3

        return result

    def simulate(self, test_state, a):
        # Backup current state and player
        snapshot = self.board.copy()
        # play on given state
        self.board = test_state.copy()
        state, reward, result = self.step(a)
        # Restore state and current player
        self.board = snapshot

        return state, reward, result

    def get_all_next_actions(self):
        return [action for action in range(actions_count)]

    # Retursns the actions in this fashion:
    # [1, 1, 1, 0, 0, 1, 1, 1, 1], meaning that actions 0, 1, 2, 5, 6, 7, 8
    # are valid but actions 3 and 4 are not
    def get_valid_actions(self, state=None):
        state = state if state is not None else self.board.copy()
        actions = []
        for a in range(actions_count):
            row_idx, col_idx = self._to_idx(a)
            if self.board[row_idx, col_idx] > 0:
                actions.append(0)
            else:
                actions.append(1)
        return actions

    def reset(self):
        self.board = np.full((self.width, self.height), 0)

    def to_str(self, board=None):
        string = os.linesep
        board = board if board is not None else self.board
        b = board.reshape(self.width * self.height)
        for idx, c in enumerate(b):
            c = int(c)
            if (idx + 1) % self.width > 0:
                string = '{}{} '.format(string, c)
            else:
                string = '{}{}{}'.format(string, c, os.linesep)
        return string

    def print(self, board=None):
        print(self.to_str(board))

    def get_state(self):
        return self.board.copy().astype(dtype=np.float32)

    def get_symmetric_around_center(self, board=None):
        board = board if board else self.board
        sym = np.full((self.width, self.height), 0)
        for i in range(self.width):
            for j in range(self.height):
                sym[i, j] = board[2 - i, 2 - j]
        return sym

    def get_rot90_clockwise(self, board=None):
        board = board if board else self.board
        im = np.full((self.width, self.height), 0)
        for i in range(self.width):
            for j in range(self.height):
                im[i, j] = board[2 - j, i]
        return im

    def get_rot90_counterclockwise(self, board=None):
        board = board if board else self.board
        im = np.full((self.width, self.height), 0)
        for i in range(self.width):
            for j in range(self.height):
                im[i, j] = board[j, 2 - i]
        return im

    def get_sym_main_diag(self, board=None):
        board = board if board else self.board
        im = board.copy()
        return im.T

    def get_sym_second_diag(self, board=None):
        board = board if board else self.board
        im = np.full((self.width, self.height), 0)
        for i in range(self.width):
            for j in range(self.height):
                im[i, j] = board[2 - j, 2 - i]
        return im

    def get_sym_horizontal_axis(self, board=None):
        board = board if board else self.board
        im = np.full((self.width, self.height), 0)
        for i in range(self.width):
            for j in range(self.height):
                im[i, j] = board[2 - i, j]
        return im

    def get_sym_vertical_axis(self, board=None):
        board = board if board else self.board
        im = np.full((self.width, self.height), 0)
        for i in range(self.width):
            for j in range(self.height):
                im[i, j] = board[i, 2 - j]
        return im



    # Returns the inverse of the current state (1s are 2s and 2s are 1s)
    def get_inv_state(self, board=None):
        board = board if board is not None else self.board
        inv = board.copy()
        for col_idx in range(self.width):
            for row_idx in range(self.height):
                if inv[col_idx, row_idx] == 1:
                    inv[col_idx, row_idx] = 2
                elif inv[col_idx, row_idx] == 2:
                    inv[col_idx, row_idx] = 1
        return inv


def main():
    b = TicTacToeEnv()
    # for a in [1, 0, 2, 3, 6, 7, 4]:
    #     b.step(a)
    # b.print()
    #
    # b.print(b.get_rot90_clockwise())
    # b.print(b.get_rot90_counterclockwise())
    # b.print(b.get_sym_horizontal_axis())
    # b.print(b.get_sym_vertical_axis())
    # b.print(b.get_sym_main_diag())
    # b.print(b.get_sym_second_diag())
    # b.print(b.get_symmetric_around_center())

    while True:
        player = b.get_current_player()
        b.print()
        action = int(input('Player {}\'s turn. Please input the col number (0 to {}) you want to place your chip: '.format(player, actions_count - 1)))
        state, reward, result = b.step(action)
        if result < 0:
            print('Your input is invalid.')
        elif result == 0:
            pass
        elif result == 3:
            print('Draw game!!!')
            break
        else:
            print('Player', player, 'won!!!')
            b.print()
            break


if __name__ == '__main__':
    main()
