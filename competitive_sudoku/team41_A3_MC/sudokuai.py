from copy import deepcopy
from math import log, sqrt
from random import choice, shuffle
from time import time
from itertools import product

from competitive_sudoku.sudoku import Move
import competitive_sudoku.sudokuai


global_C = 2
global_total = False
global_selection = "max" #max or robust


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()
    
    def compute_best_move(self, game_state) -> None:
        """
        The AI calculates the best move for the given game state.
        It continuously updates the best_move value until forced to stop.
        Firstly, it creates a solved version of the sudoku such that it has a valid number for every square.
        Then it repeatedly uses minimax to determine the best square, at increasing depths.
        @param game_state: The starting game state.
        """
        start_time = time()

        # Create some global variables so we do not have to pass these around
        global global_m, global_n, global_N
        global_m = game_state.board.m
        global_n = game_state.board.n
        global_N = global_m*global_n

        # Calculates the starting variable we will be using
        possible_moves, numbers_left, empty_squares = get_possible_numbers_and_empty(game_state.board)

        # Quickly propose a valid move to have something to present
        self.quick_propose_valid_move(game_state, possible_moves, numbers_left)

        # If this is the first turn we are in control, we figure out and save the alloted time
        available_time = self.load()
        if available_time == None:
            while True:
                cur_time = time()-start_time
                self.save(cur_time)
        
        # Gives a solution of the board
        solved_board = solve_sudoku(deepcopy(game_state.board))

        # Set the starting score of the MC_tree
        if not global_total: score = game_state.scores[0] - game_state.scores[1]
        else: score = 0

        # The root of the MC_tree
        root = Node(None, 1, possible_moves, empty_squares, score, 0)

        stop_time = available_time*0.95 - 0.001
        # We will loop this until we get low on time
        while time()-start_time < stop_time:
            one_mc_loop(root)
            
        # Determine the best child at this point
        max_value = -float('inf')
        move_index = None
        if global_selection == "max":
            for child in root.children:
                if child.q/child.n > max_value:
                    max_value = child.q/child.n
                    move_index = child.index
        
        elif global_selection == "robust":
            for child in root.children:
                if child.n > max_value:
                    max_value = child.n
                    move_index = child.index

        # Propose the current best move
        move = root.possible_moves[move_index]
        number_to_use = solved_board.get(move[0], move[1])
        self.propose_move(Move(move[0], move[1], number_to_use))


    def quick_propose_valid_move(self, game_state, possible_moves: list, numbers_left: dict) -> None:
        """
        Proposes a random move which is neither illegal, nor a taboo move, though it might be a bomb move.
        @param game_state: The game state for which this is happening.
        @param possible_moves: A list of coordinates for all empty squares (result of the get_possible_moves function).
        @param numbers_left: A dictionary with for each group which number are not in that group (result of the get_numbers_left function).
        """

        move = choice(possible_moves)
        numbers = set(numbers_left["rows"][move[0]] & numbers_left["columns"][move[1]] & numbers_left["regions"][int(move[0] \
            / game_state.board.m)*game_state.board.m + int(move[1] / game_state.board.n)])
        moves = [Move(move[0], move[1], number) for number in numbers]
        for move in moves:
            if move not in game_state.taboo_moves:
                self.propose_move(move)
                break

def one_mc_loop(root) -> None:
    """
    Do one single loop of the the MC_tree algorithm.
    @param root: The root node of the tree to work on.
    """

    # Selection Step
    cur_node = root
    while cur_node.children:
        cur_node = cur_node.selection_step()
        
    # Expansion Step
    if cur_node.n == 1:
        cur_node = cur_node.add_children()
    
    # Simulation Step
    result = cur_node.simulate()
    
    if not global_total:
        if result > 0:
            result = 1
        elif result < 0:
            result = -1
    
    # Backpropagation Step
    cur_node.send_added_q(result)

class Node():
    def __init__(self, parent, starting: int, possible_moves: list, empty_squares: list, score: int, index: int):
        self.n = 0 # Amount of times this node was visited
        self.q = 0 # Total score at this node
        self.UCT = float('inf') # Upper Confidence Bound for Trees
        
        self.parent = parent
        
        self.starting = starting
        self.possible_moves = possible_moves
        self.empty_squares = empty_squares
        self.score = score
        
        self.children = [] 
        self.children_UCT = [] # This will keep track of the children's UCT values
        
        self.index = index
    
    def __str__(self) -> str:
        return f'n:{self.n}, q:{self.q}, UCT:{self.UCT}, score:{self.score}'
    
    def selection_step(self):
        """
        Returns the child with the highest UCT.
        """
        return self.children[max(range(len(self.children_UCT)), key=self.children_UCT.__getitem__)]
    
    def add_children(self):
        """
        For each move possible in this node create a child.
        @return: Returns a random child.
        """
        self.children = [None]*len(self.possible_moves)
        
        # For each move a child
        for new_move in self.possible_moves:
            
            # Update the amount of points earned by this move
            points = (self.empty_squares[new_move[0]] == 1) + \
            (self.empty_squares[new_move[1]+global_N] == 1) + \
            (self.empty_squares[square2region(new_move)+global_N*2] == 1)

            new_score = self.score + self.starting*{0:0, 1:1, 2:3, 3:7}[points]

            # Flip wether the node is the starting player
            new_starting = -self.starting
            
            # Make copies of possible_moves and empty_squares and update them
            new_possible_moves = self.possible_moves.copy()
            new_possible_moves.remove(new_move)

            new_empty_squares = self.empty_squares.copy()
            new_empty_squares[new_move[0]] -= 1
            new_empty_squares[new_move[1]+global_N] -= 1
            new_empty_squares[square2region(new_move)+global_N*2] -= 1

            # Send the index of the child in this parent node for easy bookkeeping
            new_index = self.possible_moves.index(new_move)

            # Create the child
            new_child = Node(self, new_starting, new_possible_moves, new_empty_squares, new_score, new_index)
            self.children[new_index] = new_child
        
        # Setups the children UCT storage list
        self.children_UCT = [float('inf')]*len(self.possible_moves)
        
        # Return a random child
        if self.children:
            return choice(self.children)
        else:
            return self     
    
    def send_added_q(self, added_q: int) -> None:
        """
        Updates the n and q in this node and sends the function upwards the tree.
        @param added_q: The amount that should be added to q (influenced by 'starting').
        """
        self.n += 1
        self.q += self.starting*added_q
        
        if self.parent != None: # Only do this if we are not at the root
            self.UCT = (self.q/self.n) + global_C*sqrt(log(self.parent.n+1)/self.n)
            self.parent.children_UCT[self.index] = self.UCT
            self.parent.send_added_q(added_q)
    
    def simulate(self):
        """
        Simulates a game from this node by randomly picking moves.
        """
        
        # Create copies of all relevant values that we can change
        cur_starting = self.starting
        cur_score = self.score
        cur_possible_moves = self.possible_moves.copy()
        cur_empty_squares = self.empty_squares.copy()
        
        # Loop over a shuffled list to simulate random moves
        shuffle(cur_possible_moves)
        while cur_possible_moves:
            move = cur_possible_moves.pop()
            
            # Get and update points
            points = (self.empty_squares[move[0]] == 1) + \
            (self.empty_squares[move[1]+global_N] == 1) + \
            (self.empty_squares[square2region(move)+global_N*2] == 1)

            cur_score = cur_score + cur_starting*{0:0, 1:1, 2:3, 3:7}[points]
            
            # Update empty_squares
            cur_empty_squares[move[0]] -= 1
            cur_empty_squares[move[1]+global_N] -= 1
            cur_empty_squares[square2region(move)+global_N*2] -= 1
            
            # Flip wether we are currently the starting player
            cur_starting = -cur_starting
        
        return cur_score


def square2region(square: tuple) -> int:
    """
    From the coordinates of a square return the region the square is in.
    @param square: the x and y coordinates of a square.
    """
    region_number = square[0] - square[0] % global_m
    region_number += square[1]  // global_n
    return(region_number)


def get_possible_numbers_and_empty(board) -> set:
    """
    For the current board get the possible_moves, numbers_left and empty_squares.
    Possible_moves: The coordinates of all square that are still empty.
    Numbers_left: The numbers not yet in a group for each row/column/region
    Empty_squares: The number of empty (zero) squares for each row/column/region.
    Note: read 'possible, numbers and empty'.
    @param board: The board this should be done on.
    @return: A set with the possible_moves list, the numbers_left dictionary and the empty_squares dictionary.
    """
    
    # The variables we will be adding to while looping
    possible_moves = []
    numbers_present = {"rows": [[] for i in range(board.N)], "columns": [[] for i in range(board.N)], "regions": [[] for i in range(board.N)]}
    empty_squares = [0]*(3*board.N)

    # Loop over every square
    for row in range(board.N):        
        for column in range(board.N):
            region = square2region((row,column))
            
            value = board.get(row, column)
            empty = (value == 0)
            
            if empty:
                possible_moves.append((row,column))
                
            numbers_present["rows"][row].append(value)
            empty_squares[row] += empty
            
            numbers_present["columns"][column].append(value)
            empty_squares[column+board.N] += empty
            
            numbers_present["regions"][region].append(value)
            empty_squares[region+board.N*2] += empty
    
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