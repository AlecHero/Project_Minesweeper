import torch
import itertools

# Define game parameters
ROWS = 5
COLS = 5
MINES = 5

# Define the states
CLOSED = 9
MINE = -1

# Define the actions
ACTIONS = ROWS * COLS

force_cpu = True

if not force_cpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# place mines
def place_mines(mine_board):
    mines_placed = 0

    while mines_placed < MINES:
        rnd = torch.randint(0, ROWS * COLS, (1,)).to(device)
        x = int(rnd // ROWS)
        y = int(rnd % COLS)
        if mine_board[x, y] != MINE:
            mine_board[x, y] = MINE
            mines_placed += 1

    return mine_board


# make last layer all closed
def reset_board(board):

    board = torch.zeros((10, ROWS, COLS), dtype=torch.int32).to(device)
    mine_board = torch.zeros((ROWS, COLS), dtype=torch.int32).to(device)
    board[9] = 1

    mine_board = place_mines(mine_board)

    return board, mine_board


# check for mines in the neighbours
def count_neighbour_mines(mine_board, x, y):
    return sum(
        _x >= 0
        and _x < ROWS
        and _y >= 0
        and _y < COLS
        and mine_board[_x, _y] == MINE
        for _x, _y in itertools.product(
            range(x - 1, x + 2), range(y - 1, y + 2)
        )
    )
                    

# open the neighbours
def open_neighbour_cells(board, mine_board, x, y):
    game_over = False
    new = False

    if board[CLOSED, x, y] == 0:
        new = False
        return board, game_over, new

    elif board[CLOSED, x, y] == 1:
        new = True

        if mine_board[x, y] == MINE:
            game_over = True
            return board, game_over, new

        board[CLOSED, x, y] = 0

        mines = count_neighbour_mines(mine_board, x, y)
        board[mines, x, y] = 1

        if mines == 0:
            for _x in range(x-1, x+2):
                for _y in range(y-1, y+2):
                    if (
                        _x >= 0
                        and _x < ROWS
                        and _y >= 0
                        and _y < COLS
                        and board[CLOSED, _x, _y] == 1
                    ):
                        mines = count_neighbour_mines(mine_board, _x, _y)
                        if mines == 0:
                            board, _, _ = open_neighbour_cells(board, mine_board, _x, _y)
                        else:
                            board[mines, _x, _y] = 1
                            board[CLOSED, _x, _y] = 0    

    return board, game_over, new


# Define the boards
board = torch.zeros((10, ROWS, COLS), dtype=torch.int32).to(device)
mine_board = torch.zeros((ROWS, COLS), dtype=torch.int32).to(device)

def step(board, action):
    x = int(action // ROWS)
    y = int(action % COLS)

    board, game_over, new = open_neighbour_cells(board, mine_board, x, y)

    if game_over:
        reward = -100
        done = True

    elif torch.sum(board[CLOSED]) == MINES:
        reward = 2500
        done = True
    
    elif not new:
        reward = -10
        done = False

    else:
        reward = 50
        done = False

    return board, reward, done

# import pygame
# import itertools

# for x, y in itertools.product(range(ROWS), range(COLS)):
#     if board[CLOSED, x, y] == 1:
#         pygame.draw.rect(screen, (144, 238, 144), (pp + x * square_size, pp + y * square_size, square_size, square_size))
#     elif board[0, x, y] == 1:
#         pygame.draw.rect(screen, (255, 255, 255), (pp + x * square_size, pp + y * square_size, square_size, square_size))
#     else:
#         pygame.draw.rect(screen, (255, 255, 255), (pp + x * square_size, pp + y * square_size, square_size, square_size))
#         text = str(game_board[x, y].item())
#         font = pygame.font.SysFont('Arial', 30)
#         text_surface = font.render(text, True, (0, 0, 0))
#         screen.blit(text_surface, (pp + x * square_size + 20, pp + y * square_size + 10))

#     if mine_board[x, y] == -1:
#         pygame.draw.rect(screen, (255, 0, 0), (pp + x * square_size + 10, pp + y * square_size + 10, square_size - 20, square_size - 20))

#     if x == int(action // ROWS) and y == int(action % COLS):
#         pygame.draw.rect(screen, (0, 0, 255), (pp + x * square_size + 5, pp + y * square_size + 5, 10, 10))

# # render the grid
# for x in range(ROWS + 1):
#     pygame.draw.line(screen, (0, 0, 0), (pp + x * square_size, pp), (pp + x * square_size, pp + COLS * square_size))
# for y in range(COLS + 1):
#     pygame.draw.line(screen, (0, 0, 0), (pp, pp + y * square_size), (pp + ROWS * square_size, pp + y * square_size))


# action_line_enabled = True
# one_hot_board = False
# slow_mode = False

# # draw a line from each action to the next
# if action_line_enabled:
#     action_line.append(action)
#     for i in range(len(action_line) - 1):
#         pygame.draw.line(screen, (0, 0, 255), (pp + int(action_line[i] // ROWS) * square_size + 5, pp + int(action_line[i] % COLS) * square_size + 5), (pp + int(action_line[i + 1] // ROWS) * square_size + 5, pp + int(action_line[i + 1] % COLS) * square_size + 5))

# if one_hot_board:
#     # draw the 10 small boards from one hot encoding 5 wide and 2 tall to the right of the big board start from the top left corner
#     for i in range(10):
#         for x, y in itertools.product(range(ROWS), range(COLS)):
#             if board[i, x, y] == 1:
#                 pygame.draw.rect(screen, (102, 153, 204), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + x * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size, small_square_size, small_square_size))
#             else:
#                 pygame.draw.rect(screen, (255, 255, 255), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + x * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size, small_square_size, small_square_size))

#         # render the grid
#         for x in range(ROWS + 1):
#             pygame.draw.line(screen, (0, 0, 0), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + x * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + x * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + COLS * small_square_size))
#         for y in range(COLS + 1):
#             pygame.draw.line(screen, (0, 0, 0), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + ROWS * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size))


# pygame.display.update()
# if slow_mode:
#     pygame.time.delay(2000)

