import sys
import random
import time
from copy import deepcopy
import numpy as np
from mpi4py import MPI

class GameBoard:
    EMPTY_SLOT = '-'
    HUMAN_PLAYER = 'x'
    CPU_PLAYER = 'o'

    def __init__(self, num_rows=6, num_cols=7):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.board_grid = None
        self.column_heights = None
        self.last_player = None
        self.last_col_played = None
        self.initialize_board()

    def clear_board(self):
        if self.board_grid is None:
            return
        self.board_grid = None
        self.column_heights = None

    def initialize_board(self):
        self.board_grid = np.full((self.num_rows, self.num_cols), self.EMPTY_SLOT)
        self.column_heights = np.zeros(self.num_cols, dtype=int)

    def __deepcopy__(self, memo):
        new_board = GameBoard(self.num_rows, self.num_cols)
        new_board.last_player = self.last_player
        new_board.last_col_played = self.last_col_played
        new_board.board_grid = np.copy(self.board_grid)
        new_board.column_heights = np.copy(self.column_heights)
        return new_board

    def is_move_legal(self, column):
        assert column < self.num_cols
        return self.board_grid[self.num_rows - 1, column] == self.EMPTY_SLOT

    def make_move(self, column, player):
        if not self.is_move_legal(column):
            return False
        self.board_grid[self.column_heights[column], column] = player
        self.column_heights[column] += 1
        self.last_player = player
        self.last_col_played = column
        return True

    def undo_move(self, column):
        assert column < self.num_cols
        if self.column_heights[column] == 0:
            return False
        self.column_heights[column] -= 1
        self.board_grid[self.column_heights[column], column] = self.EMPTY_SLOT
        return True

    def is_game_over(self, column):
        assert column < self.num_cols
        row = self.column_heights[column] - 1
        if row < 0:
            return False
        player = self.board_grid[row, column]

        seq_count = 1
        r = row - 1
        while r >= 0 and self.board_grid[r, column] == player:
            seq_count += 1
            r -= 1
        if seq_count > 3:
            return True
        
        seq_count = 0
        c = column
        while c > 0 and self.board_grid[row, c - 1] == player:
            c -= 1
        while c < self.num_cols and self.board_grid[row, c] == player:
            seq_count += 1
            c += 1
        if seq_count > 3:
            return True

        seq_count = 0
        r, c = row, column
        while c > 0 and r > 0 and self.board_grid[r - 1, c - 1] == player:
            c -= 1
            r -= 1
        while c < self.num_cols and r < self.num_rows and self.board_grid[r, c] == player:
            seq_count += 1
            c += 1
            r += 1
        if seq_count > 3:
            return True

        seq_count = 0
        r, c = row, column
        while c > 0 and r < self.num_rows - 1 and self.board_grid[r + 1, c - 1] == player:
            c -= 1
            r += 1
        while c < self.num_cols and r >= 0 and self.board_grid[r, c] == player:
            seq_count += 1
            c += 1
            r -= 1
        if seq_count > 3:
            return True

        return False

    def load_board(self, filename):
        with open(filename, 'r') as file:
            self.num_rows, self.num_cols = map(int, file.readline().split())
            self.clear_board()
            self.initialize_board()
            for r in range(self.num_rows - 1, -1, -1):
                line = file.readline().split()
                for c in range(self.num_cols):
                    self.board_grid[r, c] = (line[c])
            for c in range(self.num_cols):
                for h in range(self.num_rows):
                    if self.board_grid[h, c] == self.EMPTY_SLOT:
                        break
                self.column_heights[c] = h
        return True

    def save_board(self, filename):
        with open(filename, 'w') as file:
            file.write(f"{self.num_rows} {self.num_cols}\n")
            for i in range(self.num_rows - 1, -1, -1):
                for j in range(self.num_cols):
                    file.write(f" {self.board_grid[i, j]} ")
                file.write("\n")
    
    def display_board(self):
        for r in range(self.num_rows - 1, -1, -1):
            print(' '.join(self.board_grid[r]))
        print()

SEARCH_DEPTH = 7

def evaluate_board(current_board, current_player, last_move_col, depth_remaining):
    if current_board.is_game_over(last_move_col):
        if current_player == GameBoard.CPU_PLAYER:
            return 1
        else:
            return -1

    if depth_remaining == 0:
        return 0

    depth_remaining -= 1
    next_player = GameBoard.HUMAN_PLAYER if current_player == GameBoard.CPU_PLAYER else GameBoard.CPU_PLAYER
    total_score = 0
    possible_moves = 0
    all_lose = True
    all_win = True

    for col in range(current_board.num_cols):
        if current_board.is_move_legal(col):
            possible_moves += 1
            current_board.make_move(col, next_player)
            move_result = evaluate_board(current_board, next_player, col, depth_remaining)
            current_board.undo_move(col)

            if move_result > -1:
                all_lose = False
            if move_result != 1:
                all_win = False
            if move_result == 1 and next_player == GameBoard.CPU_PLAYER:
                return 1
            if move_result == -1 and next_player == GameBoard.HUMAN_PLAYER:
                return -1

            total_score += move_result

    if all_win:
        return 1
    if all_lose:
        return -1

    return total_score / possible_moves

def worker_process(comm):
    while True:
        task_data = comm.recv(source=0)
        if task_data == 'DONE':
            break

        board_state, depth, columns_to_evaluate = task_data
        eval_results = []
        for col in columns_to_evaluate:
            if board_state.is_move_legal(col):
                board_state.make_move(col, GameBoard.CPU_PLAYER)
                eval_result = evaluate_board(deepcopy(board_state), GameBoard.CPU_PLAYER, col, depth - 1)
                board_state.undo_move(col)
                eval_results.append((eval_result, col))
        comm.send(eval_results, dest=0)

def get_human_move(board):
    board.display_board()
    chosen_move = -1
    while chosen_move < 0 or chosen_move >= board.num_cols or not board.is_move_legal(chosen_move):
        chosen_move = int(input("Enter move please, it has to be between 0 and 6: "))
    return chosen_move 

def create_tasks(board, depth, depth_level):
    task_list = []
    if depth_level == 1:
        for col in range(board.num_cols):
            if board.is_move_legal(col):
                new_board = deepcopy(board)
                new_board.make_move(col, GameBoard.CPU_PLAYER)
                task_list.append((new_board, GameBoard.CPU_PLAYER, col, depth - 1))
    else:
        for col in range(board.num_cols):
            if board.is_move_legal(col):
                new_board = deepcopy(board)
                new_board.make_move(col, GameBoard.CPU_PLAYER)
                sub_tasks = create_tasks(new_board, depth - 1, depth_level - 1)
                for sub_task in sub_tasks:
                    task_list.append(sub_task)
    return task_list

def distribute_tasks(task_list, num_workers):
    chunk_size = max(1, len(task_list) // num_workers)
    return [task_list[i:i + chunk_size] for i in range(0, len(task_list), chunk_size)]

comm = MPI.COMM_WORLD
process_rank = comm.Get_rank()
num_processes = comm.Get_size()

if process_rank == 0:
    if len(sys.argv) < 3:
        print("Not enough args given. Two args needed -> board(.txt) and depth-serial(int)")
        exit(0)

    board_filename = sys.argv[1]
    serial_depth = int(sys.argv[2])

    game_board = GameBoard()
    game_board.load_board(board_filename)
    search_depth = SEARCH_DEPTH

    for col in range(game_board.num_cols):
        game_board.is_game_over(col)
        if game_board.is_game_over(col):
            print("Game over!")
            for i in range(1, num_processes):
                comm.send('DONE', dest=i)
            exit(0)

    while True:
        human_move = get_human_move(game_board)
        game_board.make_move(human_move, GameBoard.HUMAN_PLAYER)

        start_time = time.time()
        if game_board.is_game_over(human_move):
            game_board.display_board()
            print("Human wins!")
            for i in range(1, num_processes):
                comm.send('DONE', dest=i)
            break

        tasks = create_tasks(game_board, SEARCH_DEPTH, serial_depth)
        
        if num_processes > 1:
            task_chunks = distribute_tasks(tasks, num_processes - 1)

            for i, chunk in enumerate(task_chunks):
                if i < num_processes - 1:
                    comm.send(chunk, dest=i + 1)

            local_best_score = -1
            local_best_col = -1

            for task in task_chunks[0]:
                board_state, last_player, col, task_depth = task
                result = evaluate_board(board_state, last_player, col, task_depth)
                if result > local_best_score or (result == local_best_score and random.random() < 0.5):
                    local_best_score = result
                    local_best_col = col

            final_results = [(local_best_score, local_best_col)]
            for _ in range(1, num_processes):
                worker_results = comm.recv()
                final_results.extend(worker_results)

            best_score = -1
            best_col = -1

            for result, col in final_results:
                if result > best_score or (result == best_score and random.random() < 0.5):
                    best_score = result
                    best_col = col
        else:
            local_best_score = -1
            best_col = -1
            for task in tasks:
                board_state, last_player, col, task_depth = task
                result = evaluate_board(board_state, last_player, col, task_depth)
                if result > local_best_score or (result == local_best_score and random.random() < 0.5):
                    local_best_score = result
                    best_col = col
            best_score = local_best_score

        if best_score == -1:
            search_depth //= 2
        else:
            game_board.make_move(best_col, GameBoard.CPU_PLAYER)
            game_board.save_board("board.txt")
            for col in range(game_board.num_cols):
                if game_board.is_game_over(col):
                    game_board.display_board()
                    print("Game over! (CPU wins)")
                    for i in range(1, num_processes):
                        comm.send('DONE', dest=i)
                    end_time = time.time()
                    print("{} seconds".format(end_time - start_time))
                    exit(0)
            print("Best column: {}, best value: {}".format(best_col, best_score))
            end_time = time.time()
            print("{} seconds".format(end_time - start_time))
else:
    while True:
        task_list = comm.recv(source=0)
        if task_list == 'DONE':
            break
        eval_results = []
        for task in task_list:
            board_state, last_player, col, task_depth = task
            result = evaluate_board(board_state, last_player, col, task_depth)
            eval_results.append((result, col))
        comm.send(eval_results, dest=0)
