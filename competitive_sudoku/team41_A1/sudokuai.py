import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
import competitive_sudoku.sudokuai
from competitive_sudoku.execute import solve_sudoku #TODO this should be removed later

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    def __init__(self):
        super().__init__()
    
    def compute_best_move(self, game_state: GameState) -> None:
        open_squares = game_state.board.get_open_squares()
        empty_squares = game_state.board.get_empty_squares()
        for depth in range(1,9999):
            #alpha and beta are defined outside of the initialization, otherwise it breaks
            alpha = Reference(float('-inf'))
            beta = Reference(float("inf"))
            score, move = minimax(max_depth = depth, open_squares = open_squares, empty_squares = empty_squares, 
            m = game_state.board.m, n = game_state.board.n, alpha = alpha, beta = beta)
            number_to_use = get_number_to_use(game_state.board, move[0], move[1])
            self.propose_move(Move(move[0], move[1], number_to_use))
        
#The below class and functions so that we can create certain references in the minimax function
class Reference(object):
    def __init__(self, value): self.value = value

def greater(i: int, j: int) -> int:
    return i > j

def smaller(i: int, j: int) -> int:
    return i < j

def minimax(max_depth: int, open_squares: list, empty_squares: dict, m:int, n:int, alpha, beta,
is_maximizing_player: bool = True, current_score: int = 0): 
    """
    A version of the minimax algorithm.
    Every time we create a child we calculate how many points the move associated with that child might get us.
    TODO
    TODO (implements AB pruning)
    TODO (default values are those values for the first iteration)
    @param max_depth: TODO
    @param open_squares: TODO
    @param empty_squares: TODO
    @param m: The amount of rows per region for this board, used to calculate regions from coordinates.
    @param n: The amount of rows per region for this board, used to calculate regions from coordinates.
    @param is_maximizing_player: Wether the current player is attempting to maximize or minimize the score.
    @param current_score: TODO
    @alpha: The alpha for alpha-beta pruning.
    @beta: The beta for alpha-beta pruning.
    @return: TODO
    """
    
    # If we have hit either the maximum depth or if there are no more moves left we stop iteration
    if max_depth == 0 or not open_squares:
        return current_score, (-1,-1)

    # Switches values around depending on if the player is maximizing or not
    value, function, multiplier, AB, min_max = (float('-inf'), greater, 1, alpha, max) if is_maximizing_player else (float('inf'), smaller, -1, beta, min)
    
    # Initializes where we store the best move and its associated score of this node
    best_score = value
    best_move = open_squares[0]

    for move in open_squares[:]: 
        
        # Calculates how the move would change the score
        amount_finished = (empty_squares["row"][move[0]] == 1) + (empty_squares["column"][move[1]] == 1) + (empty_squares["region"][int(move[0] / m)*m + int(move[1] / n)] == 1)
        new_score = current_score + multiplier*{0:0, 1:1, 2:3, 3:7}[amount_finished]

        # Removes the move from open_squares and updates empty_squares to account for the move
        open_squares.remove(move)
        empty_squares["row"][move[0]] -= 1
        empty_squares["column"][move[1]] -= 1
        empty_squares["region"][int(move[0] / m)*m + int(move[1] / n)] -= 1
    
        # Goes one layer of minimax deeper
        returned_score = minimax(max_depth-1, open_squares, empty_squares, m, n, alpha, beta, not is_maximizing_player, new_score)[0]
       
        # Changes open_squares and empty_squares back to the original state
        open_squares.append(move)
        empty_squares["row"][move[0]] += 1
        empty_squares["column"][move[1]] += 1
        empty_squares["region"][int(move[0] / m)*m + int(move[1] / n)] += 1

        # If the score of this move going deeper is better, this becomes the best move with the best score
        if function(returned_score, best_score):
            best_score = returned_score
            best_move = move

            # Does the alpha-beta pruning
            AB.value = min_max(AB.value, best_score)
            if alpha.value >= beta.value:
                break
    
    return best_score, best_move

def get_open_squares(self) -> list:
    """
    For the current board, gets all square that are still empty
    @return: a list of all empty coordinates as sets
    """
    open_squares = []
    for i in range(self.N):
        for j in range(self.N):
            if self.get(i,j) == SudokuBoard.empty:
                open_squares.append((i,j))
    
    return open_squares

def get_empty_squares(self) -> dict:
    """
    For the current board, gets the amount of empty squares for each row/column/region
    @return: A dictionary with for the keys "row", "column" and "region" a list with the amount of empty squares per group
    """
    # Calculates the amount of empty squares per row
    empty_row = []
    for i in range(self.N):
        current_empty_row = 0
        for j in range(self.N):
            current_empty_row += self.get(i,j) == SudokuBoard.empty
        
        empty_row.append(current_empty_row)
    
    # Calculates the amount of empty squares per column
    empty_column = []
    for i in range(self.N):
        current_empty_column = 0
        for j in range(self.N):
            current_empty_column += (self.get(j,i) == SudokuBoard.empty)
        
        empty_column.append(current_empty_column)
    
    # Calculates the amount of empty squares per region
    empty_region = []
    for i in range(self.N):
        current_empty_region = 0
        for j in range(self.N):
            row = int(i/self.m)*self.m + int(j/self.n)
            column = (i%self.m)*self.n + (j%self.n)
            current_empty_region += (self.get(row,column) == SudokuBoard.empty)
        
        empty_region.append(current_empty_region)

    return {"row": empty_row, "column": empty_column, "region": empty_region}

#Gets a correct number for square i,j
def get_number_to_use(self, i, j) -> int:
    """
    For the current board, get a valid number for the square at (i,j)
    @param: The row of the relevant square
    @param: The column of the relevant square
    @returns: A valid number for the square"""
    #TODO actually make this without using solve_sudoku
    for number in range(1,self.N+1):
        new_board = copy.deepcopy(self)
        new_board.put(i, j, number)
        if solve_sudoku("bin\\solve_sudoku.exe", str(new_board)) == "The sudoku has a solution.":
            return number
    
    return -1

# Adds three function as methods of SudokuBoard for ease of use
SudokuBoard.get_open_squares = get_open_squares
SudokuBoard.get_empty_squares = get_empty_squares
SudokuBoard.get_number_to_use = get_number_to_use