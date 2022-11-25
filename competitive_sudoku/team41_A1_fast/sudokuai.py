import random
import time
import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove, load_sudoku, print_board
import competitive_sudoku.sudokuai
from competitive_sudoku.execute import solve_sudoku #TODO this should be removed later

class Reference(object):
    def __init__(self, value): self.value = value

#
#The current best version of minimax we have
#
def minimax(max_depth: int, open_squares: list, empty: dict, m:int, n:int, alpha, beta,
is_maximizing_player: bool = True, current_score: int = 0): 
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
        returned_score, done_move = minimax(max_depth-1, open_squares, empty, m, n, alpha, beta, not is_maximizing_player, new_score)

        #changes open_squares and empty back to the original state
        open_squares.append(move)
        empty["row"][move[0]] += 1
        empty["column"][move[1]] += 1
        empty["region"][int(move[0] / m)*m + int(move[1] / n)] += 1

        #if the score of this move going deeper is better, this becomes the best move with the best score
        if function(returned_score, best_score):
            best_score = returned_score
            best_move = move

            if is_maximizing_player: alpha = max(alpha, best_score)
            else: beta = min(beta, best_score)
            if alpha >= beta:
                break
    
    return best_score, best_move


def minimax2(max_depth: int, open_squares: list, empty: dict, m:int, n:int, 
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
        amount_finished = (empty[move[0]] == 1) + (empty[move[1] + n*m] == 1) + (empty[int(move[0] / m)*m + int(move[1] / n) + n*m] == 1)
        new_score = current_score + multiplier*{0:0, 1:1, 2:3, 3:7}[amount_finished]

        #removes the move from open_squares and updates empty
        open_squares.remove(move)
        empty[move[0]] -= 1
        empty[move[1] + n*m] -= 1
        empty[int(move[0] / m)*m + int(move[1] / n) + n*m*2] -= 1
    
        #goes one layer of minimax deeper
        returned_score, done_move = minimax2(max_depth-1, open_squares, empty, m, n, not is_maximizing_player, new_score, alpha, beta)
       
        #changes open_squares and empty back to the original state
        open_squares.append(move)
        empty[move[0]] += 1
        empty[move[1] + n*m] += 1
        empty[int(move[0] / m)*m + int(move[1] / n) + n*m*2] += 1

        #if the score of this move going deeper is better, this becomes the best move with the best score
        if function(returned_score, best_score):
            best_score = returned_score
            best_move = move

            AB.value = min_max(AB.value, best_score)
            if alpha.value >= beta.value:
                break
    
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

def get_empty2(self):
    result = []
    for i in range(self.N):
        current_empty_row = 0
        for j in range(self.N):
            current_empty_row += self.get(i,j) == SudokuBoard.empty
        
        result.append(current_empty_row)
    
    for i in range(self.N):
        current_empty_column = 0
        for j in range(self.N):
            current_empty_column += (self.get(j,i) == SudokuBoard.empty)
        
        result.append(current_empty_column)
    
    for i in range(self.N):
        current_empty_region = 0
        for j in range(self.N):
            row = int(i/self.m)*self.m + int(j/self.n)
            column = (i%self.m)*self.n + (j%self.n)
            current_empty_region += (self.get(row,column) == SudokuBoard.empty)
        
        result.append(current_empty_region)

    return result

def get_region_dictionary(self, moves):
    region_dictionary = {}
    for i in moves:
        region_dictionary[i] = int(i[0] / self.m)*self.m + int(i[1] / self.n)
    
    return region_dictionary

SudokuBoard.get_open_squares = get_open_squares
SudokuBoard.get_empty = get_empty
SudokuBoard.get_empty2 = get_empty2
SudokuBoard.get_region_dictionary = get_region_dictionary

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()
    
    def compute_best_move(self, game_state: GameState, max_depth = 9999, minimax_type = minimax) -> None:
        start = time.time()
        open_squares = game_state.board.get_open_squares()
        empty = game_state.board.get_empty()
        for depth in range(1,max_depth):
            alpha = float('-inf')
            beta = float("inf")
            score, move = minimax_type(max_depth = depth, open_squares = open_squares, empty = empty, 
            m = game_state.board.m, n = game_state.board.n, alpha = alpha, beta = beta)
            number_to_use = get_number_to_use(game_state.board, move[0], move[1])
            self.propose_move(Move(move[0], move[1], number_to_use))
            print("depth: " + str(depth))
            print("score: " + str(score))
            print("move: " + str(move[0]) + ", " + str(move[1]) + ", " + str(number_to_use))
            print("time: " + str(time.time()-start))
            print(" ")
    
    def compute_best_move2(self, game_state: GameState, max_depth = 9999) -> None:
        start = time.time()
        open_squares = game_state.board.get_open_squares()
        empty = game_state.board.get_empty2()
        for depth in range(1,max_depth):
            
            score, move = minimax2(max_depth = depth, open_squares = open_squares, empty = empty, 
            m = game_state.board.m, n = game_state.board.n)
            number_to_use = get_number_to_use(game_state.board, move[0], move[1])

            self.propose_move(Move(move[0], move[1], number_to_use))

ai = SudokuAI()
initial_board = load_sudoku("boards\\random-2x3.txt")
game_state = GameState(initial_board, copy.deepcopy(initial_board), [], [], [0, 0])
ai.compute_best_move(game_state, 6)


# new_board = copy.deepcopy(initial_board)
# new_board.put(1,4,6)
# new_state = GameState(initial_board, copy.deepcopy(new_board), [], [], [0, 0])
# ai.compute_best_move(new_state, 3)

# newer_board = copy.deepcopy(new_board)
# newer_board.put(3,5,5)
# newer_state = GameState(initial_board, copy.deepcopy(newer_board), [], [], [0, 0])
# ai.compute_best_move(newer_state, 2)
#things to test
# -splitting up the is_maximizing_player parts - is actually slower (somehow) by roughly 0.05
# -do not assign value to 'done_move' - maybe faster, alternatively doesn't matter
# -create dictionary for the regions, instead of calculating - maybe faster, but if so not by much
# -copy open squares - clearly slower than the current method
# -copy empty with deepcopy - a bit slower than the current overwrite methods
# -copy empty with .copy (requiring empty rework) - a bit slower
# -empty as a list instead of dictionary, no copying - definitely slower
# -does crashes matter (remove empty moves check) - please don't crash
# -allow skipping moves