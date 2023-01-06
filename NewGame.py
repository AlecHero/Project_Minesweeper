import itertools
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pygame

# ----------------------------------------Minesweeper game----------------------------------------
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
    
    # neighbour_mines = 0
    # for _x in range(x-1, x+2):
    #     for _y in range(y-1, y+2):
    #         if _x >= 0 and _x < ROWS and _y >= 0 and _y < COLS:
    #             if mine_board[_x, _y] == MINE:
    #                 neighbour_mines += 1
    # return neighbour_mines

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

        # if mines == 0:
        #     for _x in range(x-1, x+2):
        #         for _y in range(y-1, y+2):
        #             if  _x >= 0 and _x < ROWS and _y >= 0 and _y < COLS:
        #                 if board[CLOSED, _x, _y] == 1:
        #                     mines = count_neighbour_mines(mine_board, _x, _y)
        #                     if mines == 0:
        #                         board, _, _ = open_neighbour_cells(board, mine_board, _x, _y)
        #                     else:
        #                         board[mines, _x, _y] = 1
        #                         board[CLOSED, _x, _y] = 0

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



# Define the parameters
import itertools
GAMMA = 0.99
EPISODES = 100000
EPSILON = 1

# Define game parameters
ROWS = 5
COLS = 5
MINES = 5

# Define the states
CLOSED = 9
MINE = -1

# Define the actions
ACTIONS = ROWS * COLS

# ----------------------------------------Parameters----------------------------------------
force_cpu = True
training = True

square_size = 50
small_square_size = 20
slow_mode = True
render_pygame = False
action_line_enabled = True
one_hot_board = True
pp = 0

if one_hot_board:
    pp = 25

if not force_cpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Define the boards
board = torch.zeros((10, ROWS, COLS), dtype=torch.int32).to(device)
mine_board = torch.zeros((ROWS, COLS), dtype=torch.int32).to(device)

# Define the network
model = nn.Sequential(
    nn.Linear(ROWS * COLS * 10, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 256),
    nn.LeakyReLU(),
    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Linear(64, ACTIONS),
)

model.to(device)

print(device)
print(model)

# load the model if it exists
try:
    model.load_state_dict(torch.load("model5.pth"))
    print("Model loaded")
except FileNotFoundError:
    print("Model not found")

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Define the loss function
loss_fn = nn.MSELoss()
scores = []
losses = []
action_line = []

pygame.init()

# make space for 10 small boards on the right side of the big board in the center
if one_hot_board:
    board_width = (ROWS) * square_size * 3.5 - pp
    board_height = (COLS) * square_size + 2 * pp
else:
    board_width = (ROWS) * square_size
    board_height = (COLS) * square_size

screen = pygame.display.set_mode((board_width, board_height))

# Define the training loop
for episode in range(EPISODES):
    
    # exponential decay of epsilon
    EPSILON = EPSILON * 0.999
    EPSILON = max(EPSILON, 0.01)

    board, mine_board = reset_board(board)

    optimizer.zero_grad()

    # Flatten the board
    state = board.view(1, -1).float().to(device)

    score = 0

    while True:
        invalid_actions = torch.nonzero(board[CLOSED].view(-1) == 0)

        if torch.rand(1) < EPSILON:
            # choose a valid_action
            action = torch.randint(0, ACTIONS, (1,)).float().to(device)

            while action in invalid_actions:
                action = torch.randint(0, ACTIONS, (1,)).float().to(device)

            action = action[0]

        else:
            action = model(state)

            action[0, invalid_actions] = -float('inf')

            action = torch.argmax(action)

            action = action.float().to(device)

        # render every 100 episodes with pygame
        if episode % 1 == 0 and render_pygame:
            screen.fill((255, 255, 255))

            # undo one hot encoding and draw the board
            game_board = torch.argmax(board, dim=0)

            print(game_board)

            for x, y in itertools.product(range(ROWS), range(COLS)):
                if board[CLOSED, x, y] == 1:
                    pygame.draw.rect(screen, (144, 238, 144), (pp + x * square_size, pp + y * square_size, square_size, square_size))
                elif board[0, x, y] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), (pp + x * square_size, pp + y * square_size, square_size, square_size))
                else:
                    pygame.draw.rect(screen, (255, 255, 255), (pp + x * square_size, pp + y * square_size, square_size, square_size))
                    text = str(game_board[x, y].item())
                    font = pygame.font.SysFont('Arial', 30)
                    text_surface = font.render(text, True, (0, 0, 0))
                    screen.blit(text_surface, (pp + x * square_size + 20, pp + y * square_size + 10))

                if mine_board[x, y] == -1:
                    pygame.draw.rect(screen, (255, 0, 0), (pp + x * square_size + 10, pp + y * square_size + 10, square_size - 20, square_size - 20))

                if x == int(action // ROWS) and y == int(action % COLS):
                    pygame.draw.rect(screen, (0, 0, 255), (pp + x * square_size + 5, pp + y * square_size + 5, 10, 10))

            # render the grid
            for x in range(ROWS + 1):
                pygame.draw.line(screen, (0, 0, 0), (pp + x * square_size, pp), (pp + x * square_size, pp + COLS * square_size))
            for y in range(COLS + 1):
                pygame.draw.line(screen, (0, 0, 0), (pp, pp + y * square_size), (pp + ROWS * square_size, pp + y * square_size))

            
            # draw a line from each action to the next
            if action_line_enabled:
                action_line.append(action)
                for i in range(len(action_line) - 1):
                    pygame.draw.line(screen, (0, 0, 255), (pp + int(action_line[i] // ROWS) * square_size + 5, pp + int(action_line[i] % COLS) * square_size + 5), (pp + int(action_line[i + 1] // ROWS) * square_size + 5, pp + int(action_line[i + 1] % COLS) * square_size + 5))
            
            
            if one_hot_board:
                # draw the 10 small boards from one hot encoding 5 wide and 2 tall to the right of the big board start from the top left corner
                for i in range(10):
                    for x, y in itertools.product(range(ROWS), range(COLS)):
                        if board[i, x, y] == 1:
                            pygame.draw.rect(screen, (102, 153, 204), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + x * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size, small_square_size, small_square_size))
                        else:
                            pygame.draw.rect(screen, (255, 255, 255), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + x * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size, small_square_size, small_square_size))

                    # render the grid
                    for x in range(ROWS + 1):
                        pygame.draw.line(screen, (0, 0, 0), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + x * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + x * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + COLS * small_square_size))
                    for y in range(COLS + 1):
                        pygame.draw.line(screen, (0, 0, 0), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size), (ROWS * square_size - 110 + (i % 5 + 1) * (ROWS + 1) * small_square_size + ROWS * small_square_size, 50 + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size))
            
            
            pygame.display.update()
            if slow_mode:
                pygame.time.delay(2000)
                

        next_state, reward, done = step(board, action)

        next_state = next_state.view(1, -1).float().to(device)

        action_ = model(next_state)

        # find valid actions
        action_[0, invalid_actions] = -float('inf')

        action_ = torch.max(action_)

        # Calculate the loss
        loss = loss_fn(action, reward + GAMMA * action_)

        # Backpropagate
        loss.backward()

        # Update the weights
        optimizer.step()

        state = next_state

        print(mine_board)

        if done:
            break

        score += reward

    if action_line_enabled:
        action_line = []

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
            training = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            slow_mode = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
            slow_mode = True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            render_pygame = not render_pygame
        if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
            action_line_enabled = not action_line_enabled

    if not training:
        break

    scores.append(score)

    avg_score = np.mean(scores[-100:])

    if episode % 1 == 0:
        print(f'Episode: {episode:5}, Score: {score:5}, Avg Score: {avg_score:.2f}, Epsilon: {EPSILON:.4f}, Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model5.pth')
# plot learning curve
x = [i+1 for i in range(len(scores))]
running_avg = np.zeros(len(scores))
for i in range(len(running_avg)):
    running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
plt.plot(x, running_avg)
plt.title('Running average of previous 100 scores')
plt.show()
    



