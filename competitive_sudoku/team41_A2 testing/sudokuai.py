from copy import deepcopy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
from time import time  # not required for the actual code
from competitive_sudoku.sudoku import print_board


class SudokuAI(object):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        self.best_move = [0, 0, 0]
        self.lock = None

    def compute_best_move(self, game_state: GameState, max_depth) -> None:
        """
        The AI calculates the best move for the given game state.
        It continuously updates the best_move value until forced to stop.
        Firstly, it creates a solved version of the sudoku such that it has a valid number for every square.
        Then it repeatedly uses minimax to determine the best square, at increasing depths.
        @param game_state: The starting game state.
        """
        start = time()

        # Calculates the starting variable minimax needs
        open_squares = game_state.board.get_open_squares()
        empty_squares = game_state.board.get_empty_squares()
        numbers_left = game_state.board.get_numbers_left()

        # Gives a solution of the board
        solved_board = solve_sudoku(deepcopy(game_state.board), deepcopy(open_squares), numbers_left)

        # Calculate for every increasing depth
        for depth in range(1, max_depth):
            move = \
            minimax(max_depth=depth, open_squares=open_squares, empty_squares=empty_squares, m=game_state.board.m,
                    n=game_state.board.n)[1]
            number_to_use = solved_board.get(move[0], move[1])
            self.propose_move(Move(move[0], move[1], number_to_use))
            print("depth: " + str(depth))
            print("score: " + str(score))
            print("move: " + str(move[0]) + ", " + str(move[1]) + ", " + str(number_to_use))
            print("time: " + str(time() - start))
            print(" ")

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


# The below functions exist so that we can create certain references in the minimax function
def greater(i: int, j: int) -> int:
    return i > j


def smaller(i: int, j: int) -> int:
    return i < j


def minimax(max_depth: int, open_squares: list, empty_squares: dict, m: int, n: int,
            is_maximizing_player: bool = True, current_score: int = 0, alpha: int = float("-inf"),
            beta: int = float("inf")):
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
        return current_score, (-1, -1)

    # Switches values around depending on if the player is maximizing or not
    value, function, multiplier = (float('-inf'), greater, 1) if is_maximizing_player else (float('inf'), smaller, -1)

    # Initializes where we store the best move and its associated score of this node
    best_score = value
    best_move = open_squares[0]

    for move in open_squares[:]:

        # Calculates how the move would change the score
        amount_finished = (empty_squares["row"][move[0]] == 1) + (empty_squares["column"][move[1]] == 1) + (
                    empty_squares["region"][int(move[0] / m) * m + int(move[1] / n)] == 1)
        new_score = current_score + multiplier * {0: 0, 1: 1, 2: 3, 3: 7}[amount_finished]

        # Removes the move from open_squares and updates empty_squares to account for the move
        open_squares.remove(move)
        empty_squares["row"][move[0]] -= 1
        empty_squares["column"][move[1]] -= 1
        empty_squares["region"][int(move[0] / m) * m + int(move[1] / n)] -= 1

        # Goes one layer of minimax deeper
        returned_score = \
        minimax(max_depth - 1, open_squares, empty_squares, m, n, not is_maximizing_player, new_score, alpha, beta)[0]

        # Changes open_squares and empty_squares back to the original state
        open_squares.append(move)
        empty_squares["row"][move[0]] += 1
        empty_squares["column"][move[1]] += 1
        empty_squares["region"][int(move[0] / m) * m + int(move[1] / n)] += 1

        # If the score of this move going deeper is better, this becomes the best move with the best score
        if function(returned_score, best_score):
            best_score = returned_score
            best_move = move

            # Does the alpha-beta pruning
            if is_maximizing_player:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)
            if alpha >= beta:
                break

    return best_score, best_move


def get_open_squares(board: SudokuBoard) -> list:
    """
    For the current board, gets all squares that are still empty.
    @param board: The board this should be done on.
    @return: a list of all empty coordinates as tuples.
    """
    open_squares = []
    for i in range(board.N):
        for j in range(board.N):
            if board.get(i, j) == SudokuBoard.empty:
                open_squares.append((i, j))

    return open_squares


def get_empty_squares(board: SudokuBoard) -> dict:
    """
    For the current board, gets the number of empty squares for each row/column/region.
    @param board: The board this should be done on.
    @return: A dictionary with keys: "row", "column" and "region"; and values being lists with the number of empty squares per group.
    """
    # Calculates the number of empty squares per row
    empty_row = []
    for i in range(board.N):
        current_empty_row = 0
        for j in range(board.N):
            current_empty_row += board.get(i, j) == SudokuBoard.empty

        empty_row.append(current_empty_row)

    # Calculates the number of empty squares per column
    empty_column = []
    for i in range(board.N):
        current_empty_column = 0
        for j in range(board.N):
            current_empty_column += (board.get(j, i) == SudokuBoard.empty)

        empty_column.append(current_empty_column)

    # Calculates the number of empty squares per region
    empty_region = []
    for i in range(board.N):
        current_empty_region = 0
        for j in range(board.N):
            row = int(i / board.m) * board.m + int(j / board.n)
            column = (i % board.m) * board.n + (j % board.n)
            current_empty_region += (board.get(row, column) == SudokuBoard.empty)

        empty_region.append(current_empty_region)

    return {"row": empty_row, "column": empty_column, "region": empty_region}


def get_numbers_left(board: SudokuBoard):
    '''
    For the current board, gets the numbers not yet in a group for each row/column/region.
    @param board: The board this should be done on.
    @return: A dictionary with keys: "row", "column" and "region"; and values being lists with the numbers unused per group.
    '''
    # Calculates the missing numbers for each row
    rows = []
    for row in range(board.N):
        this_row = []
        for column in range(board.N):
            this_row.append(board.get(row, column))
        rows.append({x for x in range(1, board.N + 1) if x not in set(filter((0).__ne__, this_row))})

    # Calculates the missing numbers for each column
    columns = []
    for column in range(board.N):
        this_column = []
        for row in range(board.N):
            this_column.append(board.get(row, column))
        columns.append({x for x in range(1, board.N + 1) if x not in set(filter((0).__ne__, this_column))})

    # Calculates the missing numbers for each region
    regions = []
    for region in range(board.N):
        this_region = []
        for value in range(board.N):
            row = int(region / board.m) * board.m + int(value / board.n)
            column = (region % board.m) * board.n + (value % board.n)
            this_region.append(board.get(row, column))
        regions.append({x for x in range(1, board.N + 1) if x not in set(filter((0).__ne__, this_region))})

    return {"rows": rows, "columns": columns, "regions": regions}




def solve_sudoku(board, open_squares, numbers_left):
    '''
    Iteratively gives a solution to the given sudoku.
    First, fills in any squares where only one number is possible, then randomly guesses.
    @param open_squares: A list containing all still open squares/possible moves.
    @param numbers_left: A dictionary containing the missing numbers for each group???.
    @return: A filled board.
    '''
    # Finds all squares where only one number is possible
    result = []
    for move in open_squares:
        possibilities = set(numbers_left["rows"][move[0]] & numbers_left["columns"][move[1]] & \
                            numbers_left["regions"][int(move[0] / board.m) * board.m + int(move[1] / board.n)])
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
            numbers_left["regions"][int(i[0][0] / board.m) * board.m + int(i[0][1] / board.n)].remove(i[1])

        return solve_sudoku(board, open_squares, numbers_left)

    # If no squares have only 1 possible value to put in there, inspect the blocks
    # For each block, check if there is any number that only can be inputted into one square in that block


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
            new_numbers_left["regions"][int(move[0] / board.m) * board.m + int(move[1] / board.n)].remove(number)
            result = solve_sudoku(new_board, new_open_squares, new_numbers_left)

            if result != -1:
                return result

        # If no possible number worked, a previous guess was wrong 
        return -1

    # If the board is full, return
    return board


# Adds three function as methods of SudokuBoard for ease of use
SudokuBoard.get_open_squares = get_open_squares
SudokuBoard.get_empty_squares = get_empty_squares
SudokuBoard.get_numbers_left = get_numbers_left

from competitive_sudoku.sudoku import load_sudoku

ai = SudokuAI()
initial_board = load_sudoku("boards\\random-3x3.txt")
game_state = GameState(initial_board, deepcopy(initial_board), [], [], [0, 0])

#ai.compute_best_move(game_state, 6)

print(print_board(initial_board))


def present_only_once(wow: list, value: int) -> bool:
    '''True if the given value is unique in the given list of sets (or others). False if absent or present >=2 times'''
    check = 0
    for element in wow:
        if (value in element):
            check += 1
            if (check == 2):
                return(False)
    return(True)


def fill_in_one_region(board, region_number: int):
    '''For each square inside the region, create a set of possible numbers.
    If a number only is present in one of all the sets, fill the square and recurse
    @param board: board
    @param region_number: from 0 up to N times N - 1
    @return: '''

    # TO DO: make this only work with an individual region, not the entire board
    open_squares = board.get_open_squares()

    # For each open square on the board (destined to be "in one region"), save a set of still possible numbers there
    sets_for_open_squares = []
    for square in open_squares:
        numbers_left_row = get_numbers_left_one_group(board, group="row", group_number=square[0])
        numbers_left_column = get_numbers_left_one_group(board, group="column", group_number=square[1])
        numbers_left_region = get_numbers_left_one_group(board, group="region", group_number=region_number)
        numbers_left = numbers_left_row.intersection(numbers_left_column, numbers_left_region)
        sets_for_open_squares.append(numbers_left)

    # If a number is only present in one of the sets (i.e. can only be inputted in precisely one square of the region):
    change_made = False
    for number in range(1, board.N+1):
        for i in range(0, len(open_squares)):
            square = open_squares[i]
            if (present_only_once(sets_for_open_squares, number) and (number in sets_for_open_squares[i])):
            # Put this number as the value for that square and raise a flag about a change having been made
                board.put(square[0], square[1], number)
                change_made = True

    # If the flag "a change has been made" is on, recurse:
    if (change_made):
        fill_in_one_region(board, region_number)

    # If the flag "a change has been made" is off, terminate:


def get_numbers_left_one_group(board, group: str, group_number: int):
        '''
        @param board: a sudoku board
        @param group: pass "row", "column" or "region" to indicate what are you checking on
        @param group_number: integer from 0 to n times m minus 1
        @return: set of numbers not yet used inside the given group
        '''
        if (group == "row"):
            this_row = []
            for column in range(board.N):
                this_row.append(board.get(group_number, column))
            return({x for x in range(1, board.N + 1) if x not in set(filter((0).__ne__, this_row))})

        elif (group == "column"):
            this_column = []
            for row in range(board.N):
                this_column.append(board.get(row, group_number))
            return({x for x in range(1, board.N + 1) if x not in set(filter((0).__ne__, this_column))})

        elif (group == "region"):
            this_region = []
            for value in range(board.N):
                row = int(group_number / board.m) * board.m + int(value / board.n)
                column = (group_number % board.m) * board.n + (value % board.n)
                this_region.append(board.get(row, column))
            return({x for x in range(1, board.N + 1) if x not in set(filter((0).__ne__, this_region))})
        else:
            raise(ValueError("group parameter can only take the values row, column or region"))

# print(get_numbers_left_one_group(initial_board, "region", 3))
# fill_in_one_region(initial_board, 1)


def mapping_point_to_region_number(board, point : tuple) -> int:
    region_number = point[0] - point[0] % board.m
    region_number += point[1]//board.n
    return(region_number)


print(initial_board.get_open_squares())
print(mapping_point_to_region_number(initial_board, (5, 8)))
