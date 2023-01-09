import itertools
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import pygame

# ----------------------------------------Minesweeper game----------------------------------------
# place mines
def place_mines(mine_board):
    
    mines_placed = 0
    
    # Uncomment to the same board for each game
    # diff_boards = 0
    # rnd = np.random.randint(0, diff_boards)
    # torch.manual_seed(rnd)
    
    while mines_placed < MINES:
        rnd = np.random.randint(0, ROWS * COLS)
        x = rnd // ROWS
        y = rnd % COLS
        
        if mine_board[x, y] is not True:
            mine_board[x, y] = True
            mines_placed += 1
         
    # Uncomment to the same board for each game
    # seed = np.random.randint(0, 9999)
    # torch.manual_seed(seed)

    return mine_board


# make last layer all closed
def reset_board():

    board = np.zeros((10, ROWS, COLS), dtype=bool)
    mine_board = np.zeros((ROWS, COLS), dtype=bool)
    board[CLOSED] = True

    mine_board = place_mines(mine_board)
    
    return board, mine_board


# check for mines in the neighbours
def count_neighbour_mines(mine_board, x, y):

    min_x = max(0, x-1)
    max_x = min(ROWS-1, x+1)
    min_y = max(0, y-1)
    max_y = min(COLS-1, y+1)
    mines = mine_board[min_x:max_x+1, min_y:max_y+1]
    return np.sum(mines)
                    

# open the neighbours
def open_neighbour_cells(board, mine_board, x, y):
    board[CLOSED, x, y] = False

    mines = count_neighbour_mines(mine_board, x, y)
    board[mines, x, y] = True

    # if mines == 0:
    #     for _x in range(max(0, x-1), min(ROWS, x+2)):
    #         for _y in range(max(0, y-1), min(COLS, y+2)):
    #             if board[CLOSED, _x, _y]:
    #                 mines = count_neighbour_mines(mine_board, _x, _y)
    #                 if mines == 0:
    #                     board = open_neighbour_cells(board, mine_board, _x, _y)
    #                 else:
    #                     board[mines, _x, _y] = True
    #                     board[CLOSED, _x, _y] = False 

    return board


def step(board, action):
    x = action // ROWS
    y = action % COLS
    
    if mine_board[x, y]:
        reward = -100
        done = True
        
    else:
        board = open_neighbour_cells(board, mine_board, x, y)
        reward = 20
        done = False

    if np.sum(board[CLOSED]) == MINES:
        reward = 100
        done = True

    return board, reward, done


def drawnow():
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()
plt.ion()

# Define the parameters
GAMMA = 0.999
EPISODES = 1000000
EPSILON = 1

# Define game parameters
ROWS = 10
COLS = 10
MINES = 20

# Define the states
CLOSED = 9

# Define the actions
ACTIONS = ROWS * COLS

# ----------------------------------------Parameters----------------------------------------
force_cpu = True
training = True

BATCH_SIZE = 100

if not force_cpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

from torch.utils.data import Dataset, DataLoader

# create a buffer for state, action, new_action, reward, done
buffer_size = 100000
state_buffer = np.zeros((buffer_size, 10, ROWS, COLS), dtype=bool)
action_buffer = np.zeros(buffer_size, dtype=int)
new_state_buffer = np.zeros((buffer_size, 10, ROWS, COLS), dtype=bool)
new_action_buffer = np.zeros(buffer_size, dtype=int)
reward_buffer = np.zeros(buffer_size, dtype=int)
done_buffer = np.zeros(buffer_size, dtype=bool)

# Define the network CNN with a kernel size of 5
model = nn.Sequential(
    nn.Conv2d(10, 32, kernel_size=5, stride=1, padding=2),
    nn.Sigmoid(),
    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
    nn.Sigmoid(),
    nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
    nn.Sigmoid(), 
    nn.Flatten(),
    nn.Linear(128 * ROWS * COLS, ACTIONS) 
).to(device)

model.to(device)

print(device)
print(model)

# load the model if it exists
try:
    model.load_state_dict(torch.load("Ma.pth"))
    print("Model loaded")
except FileNotFoundError:
    print("Model not found")

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Define the loss function
loss_fn = nn.MSELoss()
scores = []
losses = []
step_count = 0
#----------------------------------------Training----------------------------------------
for episode in range(EPISODES):
    EPSILON = EPSILON * 0.999
    EPSILON = max(EPSILON, 0.001)

    score = 0
    done = False

    first_move = True
    
    state, mine_board = reset_board()
    
    action = np.random.randint(0, ACTIONS)

    first_move_done = False

    while first_move_done:
        x = action // ROWS
        y = action % COLS
        
        if mine_board[x, y] == True:
            first_move_done = True
            action = np.random.randint(0, ACTIONS)
        else:
            first_move_done = False
    
    while not done:
        step_count += 1

        if np.random.rand() < EPSILON and not first_move:
            action = np.random.randint(0, ACTIONS)

            while action in invalid_actions:
                action = np.random.randint(0, ACTIONS)
        elif not first_move:
            action = new_action

        if first_move:
            first_move = False
        
        # Take the action
        new_state, reward, done = step(state, action)
        
        # Find the new action
        invalid_actions = np.nonzero(state[CLOSED].flatten() == False)[0]
        
        with torch.no_grad():
            new_observation = model(torch.from_numpy(np.expand_dims(new_state, axis=0)).float().to(device))
            new_observation = new_observation.detach().cpu().numpy()
            
        new_observation[0, invalid_actions] = -np.inf
        new_action = np.argmax(new_observation)
        
        score += reward
        
        if done:
            valid_actions = np.nonzero(state[CLOSED].flatten() == True)[0]
            
            for i in valid_actions:
                x = i // ROWS
                y = i % COLS

                _, reward, done = step(state, i)
                
                if reward == 100:
                    action_reward = 100
                elif mine_board[x, y] == True:
                    action_reward = reward
                else:
                    action_reward = reward
                
                done = True
                
                step_count += 1
                
                # buffer to buffer at ind % buffer_size
                buffer_idx = step_count % buffer_size
                state_buffer[buffer_idx] = state
                action_buffer[buffer_idx] = action
                new_state_buffer[buffer_idx] = new_state
                new_action_buffer[buffer_idx] = new_action
                reward_buffer[buffer_idx] = reward
                done_buffer[buffer_idx] = done
                
        else:
            buffer_idx = step_count % buffer_size
            state_buffer[buffer_idx] = state
            action_buffer[buffer_idx] = action
            new_state_buffer[buffer_idx] = new_state
            new_action_buffer[buffer_idx] = new_action
            reward_buffer[buffer_idx] = reward
            done_buffer[buffer_idx] = done
            
        state = new_state

    # Train the model
    if step_count > buffer_size:
        batch_idx = np.random.choice(buffer_size, size=BATCH_SIZE)
        
        out = model(torch.tensor(state_buffer[batch_idx]).float())
        q_val = out[np.arange(BATCH_SIZE), action_buffer[batch_idx]]
        out_next = model(torch.tensor(new_state_buffer[batch_idx]).float())
        q_val_next = out_next[np.arange(BATCH_SIZE), new_action_buffer[batch_idx]]

        with torch.no_grad():
            target = torch.tensor(reward_buffer[batch_idx]).float() + GAMMA * q_val_next*(1-done_buffer[batch_idx])
        l = loss_fn(q_val, target.float())

        # Compute the gradients
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    scores.append(score)
    
    if episode % 10 == 0:
        avg_score = np.mean(scores[-100:])
        print(f"Episode {episode}, score {score}, avg score {avg_score}, eps {EPSILON:.3f}")
        
        # plt.clf()
        # plt.plot(scores)
        # plt.title(f'Episode {episode}, score {score}, avg score {avg_score}, eps {EPSILON:.3f}')
        # drawnow()

torch.save(model.state_dict(), 'Ma.pth')
# plot learning curve
x = [i+1 for i in range(len(scores))]
running_avg = np.zeros(len(scores))
for i in range(len(running_avg)):
    running_avg[i] = np.mean(scores[max(0, i-1000):(i+1)])
    
# make a trend line
z = np.polyfit(x, running_avg, 1)
p = np.poly1d(z)

# show formula for trend line on plot
plt.plot(x, running_avg)
plt.plot(x,p(x),"r")  
plt.legend([f"y={z[0]:.20f}x+{z[1]:.2f}"])
plt.title('Running average of previous 100 scores')
plt.show()

    



