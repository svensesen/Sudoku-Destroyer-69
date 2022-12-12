from copy import deepcopy
from competitive_sudoku.sudoku import GameState, SudokuBoard, Move, TabooMove 

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
        Firstly, it creates a solved version of the sudoku such that it has a valid number for every square.
        Then it repeatedly uses minimax to determine the best square, at increasing depths.
        @param game_state: The starting game state.
        """
        # Calculates the starting variable we will be using
        open_squares, numbers_left, empty_squares = get_open_numbers_and_empty(game_state.board)

        # Quickly propose a valid move to have something to present
        self.quick_propose_valid_move(game_state, open_squares, numbers_left)
        
        # Gives a solution of the board
        solved_board = solve_sudoku(deepcopy(game_state.board), deepcopy(open_squares), numbers_left)

        # This sets the dictionary odds, it returns 1 if the variable is odd and not 1 (only for this sudoku)
        global odds # A global variable so we do not have to pass it down (also, yes this is faster than x%2==0)
        odds = {1:0, 2:0}
        for i in range(3, game_state.board.N+1):
            odds[i] = int(i%2 == 1)

        # Calculate for every increasing depth
        for depth in range(1,9999):
            move = minimax(max_depth = depth, open_squares = open_squares, empty_squares = empty_squares, m = game_state.board.m, n = game_state.board.n)[1]
            number_to_use = solved_board.get(move[0], move[1])
            self.propose_move(Move(move[0], move[1], number_to_use))
    
    def propose_move(self, move: Move) -> None:
        """
        Note: This function is implemented here to save time with importing
        Updates the best move that has been found so far.
        N.B. DO NOT CHANGE THIS FUNCTION!
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
    
    def quick_propose_valid_move(self, game_state: GameState, open_squares: list, numbers_left: dict) -> None:
        """
        Proposes a move which is neither illegal, nor a taboo move, though it might be a bomb move.
        @param game_state: The game state for which this is happening
        @param open_squares: A list of coordinates for all empty squares (result of the get_open_squares function)
        @param numbers_left: A dictionary with for each group which number are not in that group (result of the get_numbers_left function)"""
        move = open_squares[0]
        numbers = set(numbers_left["rows"][move[0]] & numbers_left["columns"][move[1]] & numbers_left["regions"][int(move[0] \
            / game_state.board.m)*game_state.board.m + int(move[1] / game_state.board.n)])
        moves = [Move(move[0], move[1], number) for number in numbers]
        non_taboo_moves = [move for move in set(moves) if move not in set(game_state.taboo_moves)]
        self.propose_move(non_taboo_moves[0])


#The below two functions exist so that we can create certain references in the minimax function
def greater(i: int, j: int) -> int:
    return i > j


def smaller(i: int, j: int) -> int:
    return i < j


# From the coordinates of a square return the region the square is in
def square2region(square: tuple, m: int, n: int) -> int:
    region_number = square[0] - square[0] % m
    region_number += square[1]//n
    return(region_number)


def minimax(max_depth: int, open_squares: list, empty_squares: dict, m: int, n: int, 
is_maximizing_player: bool = True, current_score: int = 0, alpha: int = float("-inf"), beta: int = float("inf")) -> set: 
    """
    A version of the minimax algorithm implementing alpha-beta pruning.
    Every time we create a child, we calculate how many points the move associated with that child might get us.
    This calculation is done with empty_squares, while all potential moves are kept track of via open_squares.
    Variables with default values take those values during the first iteration.
    @param max_depth: The maximum depth the function is allowed to further search from its current depth.
    @param open_squares: A list containing all still open squares/possible moves.
    @param empty_squares: A dictionary containing the number of empty squares for each group.
    @param m: The number of rows per region for this board, used to calculate regions from coordinates.
    @param n: The number of columns per region for this board, used to calculate regions from coordinates.
    @param is_maximizing_player: Whether the current player is attempting to maximize or minimize the score.
    @param current_score: The score at the node we start this iteration of minimax on.
    @alpha: The alpha for alpha-beta pruning.
    @beta: The beta for alpha-beta pruning.
    @return: The score that will be reached from this node a maximum depth and the optimal next move to achieve that.
    """
    
    # If we have hit either the maximum depth or if there are no more moves left, we stop iteration
    if max_depth == 0 or not open_squares:
        return current_score, (-1,-1)

    # Switches values around depending on if the player is maximizing or not
    value, function, multiplier = (float('-inf'), greater, 1) if is_maximizing_player else (float('inf'), smaller, -1)
    
    # Initializes where we store the best move and its associated score of this node
    best_score = value
    best_move = open_squares[0]

    for move in open_squares[:]: 

        # Calculates how the move would change the score
        row_amount = empty_squares["row"][move[0]]
        column_amount = empty_squares["column"][move[1]]
        region_amount = empty_squares["region"][square2region(move, m, n)]

        # First one is the desire to finish rows, second one is the desire to make rows even/not odd
        amount_finished = (row_amount == 1) + (column_amount == 1) + (region_amount == 1)
        amount_even = odds[row_amount] + odds[column_amount] + odds[region_amount]

        new_score = current_score + multiplier*({0:0, 1:1, 2:3, 3:7}[amount_finished] + amount_even*0.01)

        # Removes the move from open_squares and updates empty_squares to account for the move
        open_squares.remove(move)
        empty_squares["row"][move[0]] -= 1
        empty_squares["column"][move[1]] -= 1
        empty_squares["region"][square2region(move, m, n)] -= 1
    
        # Goes one layer of minimax deeper
        returned_score = minimax(max_depth-1, open_squares, empty_squares, m, n, not is_maximizing_player, new_score, alpha, beta)[0]
       
        # Changes open_squares and empty_squares back to the original state
        open_squares.append(move)
        empty_squares["row"][move[0]] += 1
        empty_squares["column"][move[1]] += 1
        empty_squares["region"][square2region(move, m, n)] += 1

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


def get_open_numbers_and_empty(board: SudokuBoard) -> set:
    """
    For the current board get the open_squares, numbers_left and empty_squares.
    Open_squares: The coordinates of all square that are still empty.
    Numbers_left: The numbers not yet in a group for each row/column/region
    Empty_squares: The number of empty (zero) squares for each row/column/region.
    @param board: The board this should be done on.
    @return: A set with the open_squares list, the numbers_left dictionary and the empty_squares dictionary"""
    
    # The variables we will be adding to while looping
    open_squares = []
    numbers_present = {"rows": [[] for i in range(board.N)], "columns": [[] for i in range(board.N)], "regions": [[] for i in range(board.N)]}
    empty_squares = {"row": [0]*board.m*board.n, "column": [0]*board.m*board.n, "region": [0]*board.m*board.n,}

    # Loop over every square
    for row in range(board.N):        
        for column in range(board.N):
            region = square2region((row,column), board.m, board.n)
            
            value = board.get(row, column)
            empty = (value == SudokuBoard.empty)
            
            if empty:
                open_squares.append((row,column))
                
            numbers_present["rows"][row].append(value)
            empty_squares["row"][row] += empty
            
            numbers_present["columns"][column].append(value)
            empty_squares["column"][column] += empty
            
            numbers_present["regions"][region].append(value)
            empty_squares["region"][region] += empty
    
    numbers_left = {"rows": [], "columns": [], "regions": []}
    
    # Use the numbers that are present to calculate the numbers which are left
    for group in ["rows", "columns", "regions"]:
        for i in range(len(numbers_present["rows"])): 
            numbers_left[group].append({x for x in range(1,board.N+1) if x not in set(filter((0).__ne__, numbers_present[group][i]))})
        
    return open_squares, numbers_left, empty_squares

def solve_sudoku(board: SudokuBoard, open_squares: list, numbers_left: dict) -> SudokuBoard:
    '''
    Iteratively gives a solution to the given sudoku.
    First, fills in any squares where only one number is possible, then randomly guesses.
    @param open_squares: A list containing all still open squares/possible moves.
    @param empty_squares: A dictionary containing the missing numbers for each group.
    @return: A filled board.
    '''
    # Finds all squares where only one number is possible
    result = []
    for move in open_squares:
        possibilities = set(numbers_left["rows"][move[0]] & numbers_left["columns"][move[1]] & \
            numbers_left["regions"][int(move[0] / board.m)*board.m + int(move[1] / board.n)])
        if len(possibilities) == 1:
            number = next(iter(possibilities))

        # If this is the case, a previous guess was wrong
        elif len(possibilities) == 0:
            return -1

    # If squares can be filled in, do so and start back at the beginning
    if result != []:
        for i in result:
            board.put(i[0][0], i[0][1], i[1])
            open_squares.remove(i[0])
            numbers_left["rows"][i[0][0]].remove(i[1])
            numbers_left["columns"][i[0][1]].remove(i[1])
            numbers_left["regions"][int(i[0][0] / board.m)*board.m + int(i[0][1] / board.n)].remove(i[1])
            
        return solve_sudoku(board, open_squares, numbers_left)
    
    # If no squares can be filled in, keep making a guess until you hit a correct one
    elif board.empty in board.squares:
        iterator = iter(possibilities)
        for number in iterator:
            new_board = deepcopy(board)
            new_board.put(move[0], move[1], number)

            new_open_squares = deepcopy(open_squares)
            new_open_squares.remove(move)

            new_numbers_left = deepcopy(numbers_left)
            new_numbers_left["rows"][move[0]].remove(number)
            new_numbers_left["columns"][move[1]].remove(number)
            new_numbers_left["regions"][int(move[0] / board.m)*board.m + int(move[1] / board.n)].remove(number)
            result = solve_sudoku(new_board, new_open_squares, new_numbers_left)
                
            if result != -1:
                return result

        # If no possible number worked, a previous guess was wrong 
        return -1
    
    # If the board is full, return
    return board

# Makes moves hashable
def move_hash(self):
    return hash((self.i, self.j, self.value))

Move.__hash__ = move_hash
TabooMove.__hash__ = move_hash