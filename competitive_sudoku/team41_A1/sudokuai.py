#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        open_squares = game_state.board.get_open_squares()
        for depth in range(9999):
            score, move = minimax(board = game_state.board, max_depth = depth, open_squares = open_squares)
            number_to_use = get_number_to_use(game_state.board, move[0], move[1])
            self.propose_move(Move(move[0], move[1], number_to_use))


def minimax(board: SudokuBoard, max_depth: int, open_squares: list, is_maximizing_player: bool = True, current_score: int = 0): 
    if max_depth == 0 or board.is_finished:
        return current_score

    #switch values around depending on if the player is maximizing or not
    value, function = (float('-inf'), max) if is_maximizing_player else (float('inf'), min)

    for move in open_squares: 
        #creates copy of open squares without the move
        new_open_squares = open_squares[:]
        new_open_squares.remove(move)

        #TODO finish this

#Gets a correct number for square i,j
def get_number_to_use(board, i, j):

    #TODO make this
    pass

#checks if the board is completely full
def is_board_finished(self):
    return not (SudokuBoard.empty in self.squares)

#return all square values in a row expect for ones in the excluded column
def get_row_except(self, row, excluded):
    values = []
    for i in range(self.n):
        if i != excluded:
            values.append(self.get(row, i))
    
    return values

#return all square values in a column expect for ones in the excluded crow
def get_column_except(self, column, excluded):
    values = []
    for i in range(self.n):
        if i != excluded:
            values.append(self.get(i, column))
    
    return values

#return all square values in excluded_column/-row's region except for excluded_column/-row
def get_region_except(self, excluded_row, excluded_column):
    region_i = int(excluded_row / self.i)
    region_j = int(excluded_column / self.j)

    values = []
    for n in range(self.N):
        i = region_i*self.i + (n/self.j)
        j = region_j*self.j + (n%self.j)
        if not( i == excluded_row and j == excluded_column):
            values.append(self.get(i, j))
    
    return values

#gets how many points adding a number to a square would earn
def how_many_points_adds_square(self, i, j):
    finished = 0
    if SudokuBoard.empty not in self.get_row_except(i,j): 
        finished += 1

    if SudokuBoard.empty not in self.get_column_except(i,j): 
        finished += 1

    if SudokuBoard.empty not in self.get_region_except(i,j): 
        finished += 1
    
    return {0:0, 1:1, 2:3, 3:7}[finished]

#gets all currently open squares
def get_open_squares(self):
    open_squares = []
    for i in self.N:
        for j in self.N:
            if self.get(i,j) == SudokuBoard.empty:
                open_squares.append((i,j))
    
    return open_squares

SudokuBoard.is_finished = is_board_finished
SudokuBoard.get_row_except = get_row_except
SudokuBoard.get_column_except = get_column_except
SudokuBoard.get_region_except = get_region_except
SudokuBoard.points_square = how_many_points_adds_square
SudokuBoard.get_open_squares = get_open_squares