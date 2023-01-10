from itertools import product
from competitive_sudoku.sudoku import Move 

class SudokuAI(object):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    def __init__(self):
        self.best_move = [0, 0, 0]
        self.lock = None
    
    def compute_best_move(self, game_state) -> None:
        """
        The AI calculates the best move for the given game state.
        It continuously updates the best_move value until forced to stop.
        Firstly, it creates a solved version of the sudoku such that it has a valid number for every square.
        Then it repeatedly uses minimax to determine the best square, at increasing depths.
        @param game_state: The starting game state.
        """

        # Create some global variables so we do not have to pass these around
        global global_m, global_n, global_N
        global_m = game_state.board.m
        global_n = game_state.board.n
        global_N = global_m*global_n

        # Calculates the starting variable we will be using
        possible_moves, numbers_left, empty_squares = get_possible_numbers_and_empty(game_state.board)

        # Quickly propose a valid move to have something to present
        self.quick_propose_valid_move(game_state, possible_moves, numbers_left)
        
        # Gives a solution of the board
        solved_board = solve_sudoku(game_state.board)

        # This sets the dictionary odds, it returns 1 if the variable is odd and not 1 (only for this sudoku)
        global odds # A global variable so we do not have to pass it down (also, yes this is faster than x%2==0)
        odds = {1:0, 2:0}
        for i in range(3, game_state.board.N+1):
            odds[i] = int(i%2 == 1)

        # Calculate for every increasing depth
        for depth in range(1,9999):
            move = minimax(max_depth = depth, possible_moves = possible_moves, empty_squares = empty_squares)[1]
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
    
    def quick_propose_valid_move(self, game_state, possible_moves: list, numbers_left: dict) -> None:
        """
        Proposes a move which is neither illegal, nor a taboo move, though it might be a bomb move.
        @param game_state: The game state for which this is happening
        @param possible_moves: A list of coordinates for all empty squares (result of the get_possible_moves function)
        @param numbers_left: A dictionary with for each group which number are not in that group (result of the get_numbers_left function)"""
        move = possible_moves[0]
        numbers = set(numbers_left["rows"][move[0]] & numbers_left["columns"][move[1]] & numbers_left["regions"][int(move[0] \
            / game_state.board.m)*game_state.board.m + int(move[1] / game_state.board.n)])
        moves = [Move(move[0], move[1], number) for number in numbers]
        for move in moves:
            if move not in game_state.taboo_moves:
                self.propose_move(move)
                break


#The below two functions exist so that we can create certain references in the minimax function
def greater(i: int, j: int) -> int:
    return i > j


def smaller(i: int, j: int) -> int:
    return i < j


# From the coordinates of a square return the region the square is in
def square2region(square: tuple) -> int:
    region_number = square[0] - square[0] % global_m
    region_number += square[1]// global_n
    return(region_number)


def minimax(max_depth: int, possible_moves: list, empty_squares: dict, 
is_maximizing_player: bool = 1, current_score: int = 0, alpha: int = float("-inf"), beta: int = float("inf")) -> set: 
    """
    A version of the minimax algorithm implementing alpha-beta pruning.
    Every time we create a child, we calculate how many points the move associated with that child might get us.
    This calculation is done with empty_squares, while all potential moves are kept track of via possible_moves.
    Variables with default values take those values during the first iteration.
    @param max_depth: The maximum depth the function is allowed to further search from its current depth.
    @param possible_moves: A list containing all still open squares/possible moves.
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
    if max_depth == 0 or not possible_moves:
        return current_score, (-1,-1)

    # Switches values around depending on if the player is maximizing or not
    value, function, multiplier = (float('-inf'), greater, 1) if is_maximizing_player else (float('inf'), smaller, -1)
    
    # Initializes where we store the best move and its associated score of this node
    best_score = value
    best_move = possible_moves[0]

    for move in possible_moves[:]: 

        # Calculates how the move would change the score
        row_amount = empty_squares["row"][move[0]]
        column_amount = empty_squares["column"][move[1]]
        region_amount = empty_squares["region"][square2region(move)]

        # First one is the desire to finish rows, second one is the desire to make rows even/not odd
        amount_finished = (row_amount == 1) + (column_amount == 1) + (region_amount == 1)
        amount_even = odds[row_amount] + odds[column_amount] + odds[region_amount]

        new_score = current_score + multiplier*({0:0, 1:1, 2:3, 3:7}[amount_finished] + amount_even*0.01)

        # Removes the move from possible_moves and updates empty_squares to account for the move
        possible_moves.remove(move)
        empty_squares["row"][move[0]] -= 1
        empty_squares["column"][move[1]] -= 1
        empty_squares["region"][square2region(move)] -= 1
    
        # Goes one layer of minimax deeper
        returned_score = minimax(max_depth-1, possible_moves, empty_squares, not is_maximizing_player, new_score, alpha, beta)[0]
       
        # Changes possible_moves and empty_squares back to the original state
        possible_moves.append(move)
        empty_squares["row"][move[0]] += 1
        empty_squares["column"][move[1]] += 1
        empty_squares["region"][square2region(move)] += 1

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


def get_possible_numbers_and_empty(board) -> set:
    """
    For the current board get the possible_moves, numbers_left and empty_squares.
    Open_squares: The coordinates of all square that are still empty.
    Numbers_left: The numbers not yet in a group for each row/column/region
    Empty_squares: The number of empty (zero) squares for each row/column/region.
    @param board: The board this should be done on.
    @return: A set with the possible_moves list, the numbers_left dictionary and the empty_squares dictionary"""
    
    # The variables we will be adding to while looping
    possible_moves = []
    numbers_present = {"rows": [[] for i in range(board.N)], "columns": [[] for i in range(board.N)], "regions": [[] for i in range(board.N)]}
    empty_squares = {"row": [0]*board.m*board.n, "column": [0]*board.m*board.n, "region": [0]*board.m*board.n,}

    # Loop over every square
    for row in range(board.N):        
        for column in range(board.N):
            region = square2region((row,column))
            
            value = board.get(row, column)
            empty = (value == 0)
            
            if empty:
                possible_moves.append((row,column))
                
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
        
    return possible_moves, numbers_left, empty_squares

def solve_sudoku(board):
    """
    A Sudoku solver using Knuth's Algorithm X to solve an Exact Cover Problem.
    Credit goes out to Ali Assaf for inspiration from their version.
    https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html 
    @param board: The board to be solved
    @return board: A solved board
    """

    # Set X for the Exact Cover Problem
    N = board.m * board.n
    X = ([("roco", roco) for roco in product(range(N), range(N))] +
         [("ronu", ronu) for ronu in product(range(N), range(1, N + 1))] +
         [("conu", conu) for conu in product(range(N), range(1, N + 1))] +
         [("renu", renu) for renu in product(range(N), range(1, N + 1))])
    
    # Subsets Y for the Exact Cover Problem
    Y = {}
    for row, column, number in product(range(N), range(N), range(1, N + 1)):
        region = square2region((row, column))
        Y[(row, column, number)] = [
            ("roco", (row, column)),
            ("ronu", (row, number)),
            ("conu", (column, number)),
            ("renu", (region, number))]
    
    # Changes X such that we can use a dictionary instead linked lists
    X = reformat_X(X, Y)
    for row, column in product(range(N), range(N)):
        number = board.get(row, column)
        if number:
            solver_select(X, Y, (row, column, number))

    # Grabs the first solution, puts in in a board and returns it
    for solution in actual_solve(X, Y, []):
        for (row, column, number) in solution:
            board.put(row, column, number)
        return board

def reformat_X(X, Y):
    """
    Subfunction of sudoku solve
    Reformats the X to be usable
    """
    X = {i: set() for i in X}
    for key, value in Y.items():
        for i in value:
            X[i].add(key)
    return X


def actual_solve(X, Y, solution):
    """
    Subfunction of sudoku solve.
    Does the actual solving algorithm X.
    """
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = solver_select(X, Y, r)
            for s in actual_solve(X, Y, solution):
                yield s
            solver_deselect(X, Y, r, cols)
            solution.pop()

# Selects group for Algorithm X
def solver_select(X, Y, key):
    """
    Subfunction of sudoku solve.
    'Selects' a subset of X, which removes it.
    """
    cols = []
    for i in Y[key]:
        for j in X[i]:
            for k in Y[j]:
                if k != i:
                    X[k].remove(j)
        cols.append(X.pop(i))
    return cols

# Deselects group for Algorithm X
def solver_deselect(X, Y, key, cols):
    """
    Subfunction of sudoku solve.
    'Deselects' a subset of X, which adds it back.
    """
    for i in reversed(Y[key]):
        X[i] = cols.pop()
        for j in X[i]:
            for k in Y[j]:
                if k != i:
                    X[k].add(i)