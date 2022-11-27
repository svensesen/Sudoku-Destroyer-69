from copy import deepcopy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
from competitive_sudoku.execute import solve_sudoku #TODO this should be removed later
import random


class SudokuAI(object):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    def __init__(self):
        self.best_move = [0, 0, 0]
        self.lock = None
    
    def compute_best_move(self, game_state: GameState) -> None:
        """
        The AI calculates the best move for the given game state.
        It continuously updates the best_move value until forced to stop.
        Firstly it uses minimax to determine the best square, then determines a valid number for that square.
        @param game_state: The starting game state.
        """
        # Calculates the starting variable minimax needs
        open_squares = game_state.board.get_open_squares()
        empty_squares = game_state.board.get_empty_squares()

        # Calculate for every increasing depths
        for depth in range(1,9999):
            move = minimax(max_depth = depth, open_squares = open_squares, empty_squares = empty_squares, m = game_state.board.m, n = game_state.board.n)[1]
            number_to_use = get_number_to_use(game_state.board, move[0], move[1])
            move = Move(move[0], move[1], number_to_use)
            if not isinstance(move, TabooMove):
                self.propose_move(move))
    
    def propose_move(self, move: Move) -> None:
        """
        Updates the best move that has been found so far.
        N.B. DO NOT CHANGE THIS FUNCTION!
        This function is implemented here to save time with importing
        @param move: A move.
        """
        i, j, value = move.i, move.j, move.value
        if self.lock:
            self.lock.acquire()
        self.best_move[0] = i
        self.best_move[1] = j
        self.best_move[2] = value
        if self.lock:
            self.lock.release()
        
#The below functions exist so that we can create certain references in the minimax function
def greater(i: int, j: int) -> int:
    return i > j

def smaller(i: int, j: int) -> int:
    return i < j

def minimax(max_depth: int, open_squares: list, empty_squares: dict, m: int, n: int, 
is_maximizing_player: bool = True, current_score: int = 0, alpha: int = float("-inf"), beta: int = float("inf")): 
    """
    A version of the minimax algorithm implementing alpha beta pruning.
    Every time we create a child we calculate how many points the move associated with that child might get us.
    This calculation is done with empty_squares, while all potential moves are kept track of via open_squares
    Variables with default values take those values during the first iteration
    @param max_depth: The maximum depth the function is allowed to further search from its current depth
    @param open_squares: A list containing all still open squares/possible moves
    @param empty_squares: A dictionary containing the amount of empty square for each group
    @param m: The amount of rows per region for this board, used to calculate regions from coordinates.
    @param n: The amount of columns per region for this board, used to calculate regions from coordinates.
    @param is_maximizing_player: Wether the current player is attempting to maximize or minimize the score.
    @param current_score: The score at the node we start this iteration of minimax on
    @alpha: The alpha for alpha-beta pruning.
    @beta: The beta for alpha-beta pruning.
    @return: The score that will be reached from this node a maximum depth and the optimal next move to achieve that
    """
    
    # If we have hit either the maximum depth or if there are no more moves left we stop iteration
    if max_depth == 0 or not open_squares:
        return current_score, (-1,-1)

    # Switches values around depending on if the player is maximizing or not
    value, function, multiplier = (float('-inf'), greater, 1) if is_maximizing_player else (float('inf'), smaller, -1)
    
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
        returned_score = minimax(max_depth-1, open_squares, empty_squares, m, n, not is_maximizing_player, new_score, alpha, beta)[0]
       
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
            if is_maximizing_player: alpha = max(alpha, best_score)
            else: beta = min(beta, best_score)
            if alpha >= beta:
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
    def get_values_in_a_row(self, i:int, board : SudokuBoard):
        return([self.get(i, column) for column in range(board.N)])
    
    #returns the values of all squares (cells) constituting the row to which the [i, j] square (cell) belongs
    def get_values_in_a_column(self, i:int, board : SudokuBoard):
        return([self.get(i, row) for row in range(board.N)])
    
    #returns the values of all squares (cells) constituting the block to which the [i, j] square (cell) belongs
    def get_values_in_a_block(self, i:int, j:int, board : SudokuBoard):
        position = board.rc2f(i, j)
        positions = get_positions_in_a_block(position, board)
        positions_converted = [board.f2rc(position) for position in positions]
        return(self.get(x[0], x[1]) for x in positions_converted)
    
    def get_number_to_use(self, i:int, j:int, board : SudokuBoard):
        """First, it lists all unique numbers in the row and column and block of our cell(square)
        Secondly, it takes all possible values (1, 2, 3, ..., N) and deletes the ones that showed up already
        
        TODO: Check if sudoku remains solvable after this move. Also, test it."""
        
        rowvals = get_values_in_a_row(i, board)
        colvals = get_values_in_a_column(j, board)
        blockvals = get_values_in_a_block(i, j, board)
        if len(blockvals) == 1:
            out = (set(range(1, board.N)) - set(blockvals)).pop()
        else:
            if len(rowvals) == 1:
                out = (set(range(1, board.N)) - set(rowvals)).pop()
            if len(colvals) == 1:
                out = (set(range(1, board.N)) - set(colvals)).pop()
            '''they should be the same if sudoku makes sense'''
        
        numbers_used = set(rowvals) + set(colvals) + set(blockvals)
        numbers_available = set(range(1, board.N)) - numbers_used
        
        if len(numbers_available) == 1:
            return numbers_available.pop()
        
        #poss_moves = set([_ for _ in range(board.N+1)]) - set([0]) - numbers_used
        #check = lambda x: x//1.5 + 4 if x <= 6 else x//1.5 - 3
        
        return random.choice(numbers_available)
        '''errors = 999
        while errors > 0:
            board2 = deepcopy(board)
            opens = game_state.board2.get_open_squares()
            for square in opens:
                coords_x, coords_y = square[0], square[1]
                number = random.randint(1, 9)
                board2.put(coords_x, coords_y, number)
            for i in range(1, board.N):
                if len(get_values_in_a_row(i)) != len(set(get_values_in_a_row(i))):
                    errors += 1
                if len(get_values_in_a column(i)) != len(set(get_values_in_a column(i))):
                    errors += 1
            
            errors -= 999'''
                
    

# Adds three function as methods of SudokuBoard for ease of use
SudokuBoard.get_open_squares = get_open_squares
SudokuBoard.get_empty_squares = get_empty_squares
SudokuBoard.get_number_to_use = get_number_to_use