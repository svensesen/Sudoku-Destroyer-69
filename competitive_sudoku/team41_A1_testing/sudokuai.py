import random
import time
import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove, load_sudoku, print_board
import competitive_sudoku.sudokuai
from competitive_sudoku.execute import solve_sudoku #TODO this should be removed later

class Reference(object):
    def __init__(self, value): self.value = value

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState, max_depth = 9999) -> None:
        start = time.time()
        open_squares = game_state.board.get_open_squares()
        for depth in range(1,max_depth):
            score, move = minimax(board = game_state.board, max_depth = depth, open_squares = open_squares)
            number_to_use = get_number_to_use(game_state.board, move[0], move[1])
            self.propose_move(Move(move[0], move[1], number_to_use))
            print("depth: " + str(depth))
            print("score: " + str(score))
            print("move: " + str(move[0]) + ", " + str(move[1]) + ", " + str(number_to_use))
            print("time: " + str(time.time()-start))
            print(" ")
    
    def compute_best_move_fast(self, game_state: GameState, max_depth = 9999) -> None:
        start = time.time()
        open_squares = game_state.board.get_open_squares()
        empty = game_state.board.get_empty()
        for depth in range(1,max_depth):
            score, move = minimaxFast(max_depth = depth, open_squares = open_squares, empty = empty, 
            m = game_state.board.m, n = game_state.board.n)
            number_to_use = get_number_to_use(game_state.board, move[0], move[1])

            self.propose_move(Move(move[0], move[1], number_to_use))
            print("depth: " + str(depth))
            print("score: " + str(score))
            print("move: " + str(move[0]) + ", " + str(move[1]) + ", " + str(number_to_use))
            print("time: " + str(time.time()-start))
            print(" ")
    
    def compute_best_move_leaves(self, game_state: GameState, max_depth = 9999) -> None:
        start = time.time()
        open_squares = game_state.board.get_open_squares()
        empty = game_state.board.get_empty()
        leaves = [0]
        for depth in range(1,max_depth):
            score, move = minimaxLeaves(max_depth = depth, open_squares = open_squares, empty = empty, 
            m = game_state.board.m, n = game_state.board.n, leaves = leaves)

            if move != (-1,-1):
                number_to_use = get_number_to_use(game_state.board, move[0], move[1])
                self.propose_move(Move(move[0], move[1], number_to_use))
                print("depth: " + str(depth))
                print("score: " + str(score))
                print("move: " + str(move[0]) + ", " + str(move[1]) + ", " + str(number_to_use))
                print("time: " + str(time.time()-start))
                print(" ")

#
#The original minimax, is slower than minimaxFast, but for now kept in for the sake of potentially explaining the new algorithm
#
def minimax(board: SudokuBoard, max_depth: int, open_squares: list, is_maximizing_player: bool = True, current_score: int = 0): 
    if max_depth == 0 or not open_squares:
        return current_score, (-1,-1)

    #switch values around depending on if the player is maximizing or not
    value, function, multiplier = (float('-inf'), greater, 1) if is_maximizing_player else (float('inf'), smaller, -1)
    best_score = value
    best_move = open_squares[0]

    for move in open_squares: 
        #creates copy of open squares without the move
        new_open_squares = open_squares[:]
        new_open_squares.remove(move)

        #creates copy of board with the move
        new_board = copy.deepcopy(board) 
        new_board.put(move[0], move[1], 1)

        new_score = current_score + multiplier*board.points_square(move[0], move[1]) 
    
        returned_score, done_move = minimax(new_board, max_depth-1, new_open_squares, not is_maximizing_player, new_score)
        
        if function(returned_score, best_score):
            best_score = returned_score
            best_move = move
    
    return best_score, best_move

#
#The current best version of minimax we have
#
def minimaxFast(max_depth: int, open_squares: list, empty: dict, m:int, n:int, 
is_maximizing_player: bool = True, current_score: int = 0, alpha = Reference(float('-inf')), beta: list = Reference(float("inf"))): 
    #if we have hit either the maximum depth or if there are no more moves left we stop iteration
    if max_depth == 0 or not open_squares:
        return current_score, (-1,-1)

    #switch values around depending on if the player is maximizing or not
    value, function, multiplier, AB, min_max = (float('-inf'), greater, 1, alpha, max) if is_maximizing_player else (float('inf'), smaller, -1, beta, min)
    best_score = value
    best_move = open_squares[0]

    for move in open_squares[:]: 
        
        #calculates how the move would change the score
        amount_finished = (empty["row"][move[0]] == 1) + (empty["column"][move[1]] == 1) + (empty["region"][int(move[0] / m)*m + int(move[1] / n)] == 1)
        new_score = current_score + multiplier*{0:0, 1:1, 2:3, 3:7}[amount_finished]

        #removes the move from open_squares and updates empty
        open_squares.remove(move)
        empty["row"][move[0]] -= 1
        empty["column"][move[1]] -= 1
        empty["region"][int(move[0] / m)*m + int(move[1] / n)] -= 1
    
        #goes one layer of minimax deeper
        returned_score, done_move = minimaxFast(max_depth-1, open_squares, empty, m, n, not is_maximizing_player, new_score, alpha, beta)
       
        #changes open_squares and empty back to the original state
        open_squares.append(move)
        empty["row"][move[0]] += 1
        empty["column"][move[1]] += 1
        empty["region"][int(move[0] / m)*m + int(move[1] / n)] += 1

        #if the score of this move going deeper is better, this becomes the best move with the best score
        if function(returned_score, best_score):
            best_score = returned_score
            best_move = move

            AB.value = min_max(AB.value, best_score)
            if alpha.value >= beta.value:
                break
    
    return best_score, best_move

#
#The minimax below this implements the leave functionality, it is however slower than without, for now i kept it in for the sake of potential future testing
#
def minimaxLeaves(max_depth: int, open_squares: list, empty: dict, m:int, n:int, leaves: list,
is_maximizing_player: bool = True, current_score: int = 0, move:set = (-1,-1)): 
    #switch values around depending on if the player is maximizing or not
    value, function, multiplier = (float('-inf'), greater, -1) if is_maximizing_player else (float('inf'), smaller, 1)

    if max_depth == 0:
        amount_finished = (empty["row"][move[0]] == 0) + (empty["column"][move[1]] == 0) + (empty["region"][int(move[0] / m)*m + int(move[1] / n)] == 0)
        current_score = current_score +  multiplier*{0:0, 1:1, 2:3, 3:7}[amount_finished]
        leaves.append(current_score)
        return current_score, move
    
    elif not open_squares:
        return current_score, (-1,-1)
        
    best_score = value
    best_move = (-1,-1)

    if max_depth == 1:
        current_score = leaves.pop(0)

    for move in open_squares: 
        #creates copy of open squares without the move
        new_open_squares = open_squares[:]
        new_open_squares.remove(move)

        empty["row"][move[0]] -= 1
        empty["column"][move[1]] -= 1
        empty["region"][int(move[0] / m)*m + int(move[1] / n)] -= 1
    
        returned_score, done_move = minimaxLeaves(max_depth-1, new_open_squares, empty, m, n, leaves,
        not is_maximizing_player, current_score, move)
       
        empty["row"][move[0]] += 1
        empty["column"][move[1]] += 1
        empty["region"][int(move[0] / m)*m + int(move[1] / n)] += 1

        if function(returned_score, best_score):
            best_score = returned_score
            best_move = done_move
    
    return best_score, best_move

#return if i is greater than j
def greater(i, j):
    return i > j

#return if i is smaller than j
def smaller(i, j):
    return i < j

#Gets a correct number for square i,j
def get_number_to_use(board, i, j):
    #TODO actually make this without using solve_sudoku
    for number in range(1,board.N+1):
        new_board = copy.deepcopy(board)
        new_board.put(i, j, number)
        if solve_sudoku("bin\\solve_sudoku.exe", str(new_board)) == "The sudoku has a solution.":
            return number
    
    return -1

#checks if the board is completely full
def is_board_finished(self):
    return not (SudokuBoard.empty in self.squares)

#checks if a number a square i,j would complete a row
def completes_row(self, i, j):
    for column in range(self.N):
        if (self.get(i, column) == SudokuBoard.empty) and (column != j):
            return False
 
    return True

#checks if a number a square i,j would complete a column
def completes_column(self, i, j):
    for row in range(self.N):
        if (self.get(row, j) == SudokuBoard.empty) and (row != i):
            return False
 
    return True

#checks if a number a square i,j would complete a region
def completes_region(self, i, j):
    region_i = int(i / self.m)
    region_j = int(j / self.n)

    for square in range(self.N):
        row = region_i*self.m + int(square/self.n)
        column = region_j*self.n + (square%self.m)
        if (self.get(row, column) == SudokuBoard.empty) and not(row == i and column == j):
            return False
    
    return True

#gets how many points adding a number to a square would earn
def how_many_points_adds_square(self, i, j):
    finished = self.completes_row(i,j) + self.completes_column(i,j) + self.completes_region(i,j)    
    return {0:0, 1:1, 2:3, 3:7}[finished]

#gets all currently open squares
def get_open_squares(self):
    open_squares = []
    for i in range(self.N):
        for j in range(self.N):
            if self.get(i,j) == SudokuBoard.empty:
                open_squares.append((i,j))
    
    return open_squares

#gets the amount of empty entries for each row/column/region
def get_empty(self):
    empty_row = []
    for i in range(self.N):
        current_empty_row = 0
        for j in range(self.N):
            current_empty_row += self.get(i,j) == SudokuBoard.empty
        
        empty_row.append(current_empty_row)
    
    empty_column = []
    for i in range(self.N):
        current_empty_column = 0
        for j in range(self.N):
            current_empty_column += (self.get(j,i) == SudokuBoard.empty)
        
        empty_column.append(current_empty_column)
    
    empty_region = []
    for i in range(self.N):
        current_empty_region = 0
        for j in range(self.N):
            row = int(i/self.m)*self.m + int(j/self.n)
            column = (i%self.m)*self.n + (j%self.n)
            current_empty_region += (self.get(row,column) == SudokuBoard.empty)
        
        empty_region.append(current_empty_region)

    return {"row": empty_row, "column": empty_column, "region": empty_region}

#the eq and hash should make the board work as dictionary keys
def eq(self, other):
    return (self.m == other.m) and (self.squares == other.squares)
    
def hash(self):
    return hash(str(self.squares + [self.m]))

SudokuBoard.is_finished = is_board_finished
SudokuBoard.completes_row = completes_row
SudokuBoard.completes_column = completes_column
SudokuBoard.completes_region = completes_region
SudokuBoard.points_square = how_many_points_adds_square
SudokuBoard.get_open_squares = get_open_squares
SudokuBoard.get_empty = get_empty
SudokuBoard.__eq__ = eq
SudokuBoard.__hash__ = hash

def get_positions_in_a_block(position : int, board : SudokuBoard):
    """
    Printing the output of this function is equivalent to printing the positions of all numbers of the block we are in
    Useful when needing to check if we complete a block (region)
    
    This has been tested to work (math-wise)
    """

    M, N = board.region_height(), board.region_width() #double check if these are not actually put around
    
    k = (position//(N*M))%M
    l = position%N

    x = position - l - k*N*M

    lista = [] #this list will be keeping the positions (and will be returned)
    for n in range(N):
        for m in range(M):
            value = x + N*M*m + n
            lista.append(value)
    return(lista)
    
#returns the values of all squares (cells) constituting the row to which the [i, j] square (cell) belongs
def get_values_in_a_row(self, i:int, j:int, board : SudokuBoard):
    return([self.get(i, column) for column in range(self.N)])

#returns the values of all squares (cells) constituting the row to which the [i, j] square (cell) belongs
def get_values_in_a_column(self, i:int, j:int, board : SudokuBoard):
    return([self.get(i, row) for row in range(self.N)])

#returns the values of all squares (cells) constituting the block to which the [i, j] square (cell) belongs
def get_values_in_a_block(self, i:int, j:int, board : SudokuBoard):
    position = board.rc2f(i, j)
    positions = get_positions_in_a_block(position, board)
    positions_converted = [board.f2rc(position) for position in positions]
    return(self.get(x[0], x[1]) for x in positions_converted)

def get_number_to_use2(board, i, j):
    #TODO actually make this without using solve_sudoku
    for number in range(1,board.N+1):
        new_board = copy.deepcopy(board)
        new_board.put(i, j, number)
        if solve_sudoku("bin\\solve_sudoku.exe", str(new_board)) == "The sudoku has a solution.":
            return number
    
    return -1

#ai = SudokuAI()
#initial_board = load_sudoku("boards\\random-2x3.txt")
#game_state = GameState(initial_board, copy.deepcopy(initial_board), [], [], [0, 0])

#ai.compute_best_move(game_state, 6)
#ai.compute_best_move_fast(game_state, 6)
#ai.compute_best_move_leaves(game_state, 6)