import itertools
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pygame

from minesweeperGame import *


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


# The AI parameters
GAMMA = 0.99
EPISODES = 100000
EPSILON = 1


if one_hot_board:
    pp = 25
    

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


# make space for 10 small boards on the right side of the big board in the center
if one_hot_board:
    board_width = (ROWS) * square_size * 3.5 - pp
    board_height = (COLS) * square_size + 2 * pp
else:
    board_width = (ROWS) * square_size
    board_height = (COLS) * square_size

pygame.init()
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
    done = False

    while not done:
        invalid_actions = torch.nonzero(board[CLOSED].view(-1) == 0)
        print(invalid_actions)
    
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
        
        
        # get next action
        next_state, reward, done = step(board, action)
        
        

        # print(mine_board)

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