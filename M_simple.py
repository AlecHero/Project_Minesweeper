import numpy as np

CLOSED = 9

def create_board(rows, cols, mines):
    board = np.zeros((10, rows, cols), dtype=int)
    board[CLOSED] = 1
    
    mine_board = np.zeros(rows*cols, dtype=int) #<-- flat
    mine_board[np.random.choice(rows*cols, mines, replace=False)] = 1
    mine_board = mine_board.reshape(rows, cols) #<-- grid
    return board, mine_board


def solve_board(board):
    _, rows, cols = board.shape
    solved_board = board.copy()
    for i,j in np.argwhere(solved_board == -1):
        for k in range(-1, 2):
            for l in range(-1, 2):
                if i + k >= 0 and i + k < cols and j + l >= 0 and j + l < rows:
                    if solved_board[i + k][j + l] != -1:
                        solved_board[i + k][j + l] += 1
    return solved_board


def step(board, mines, mine_board, solved_board, action, is_first_move=False):
    x = action // board.shape[0]
    y = action %  board.shape[1]
    done = False
    
    # If first move is bomb
    if mine_board[x,y] and is_first_move:
        reward = 0
        mine_board[x,y] = 0
        mine_board[np.argmin(mine_board[0]),0] = 1
    # If move is bomb
    elif mine_board[x, y]:
        reward = -100
        done = True
    # If move is safe
    else:
        reward = 20
        board[CLOSED, x, y] = 0
        board[solved_board[x, y], x, y] = 1
    # Check if game is over
    if np.sum(board[CLOSED]) == mines:
        reward = 100
        done = True
    return board, reward, done