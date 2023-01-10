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


# Define the parameters
import itertools
GAMMA = 0.999
EPISODES = 1000000
EPSILON = 1

# Define game parameters
ROWS = 10
COLS = 10
MINES = 20

# Define the states
CLOSED = 9
MINE = -1

# Define the actions
ACTIONS = ROWS * COLS

# ----------------------------------------Parameters----------------------------------------
force_cpu = False
training = True

square_size = 50
small_square_size = 20
slow_mode = True
render_pygame = True
action_line_enabled = True
one_hot_board = False
pp = 0
BATCH_SIZE = 1

if one_hot_board:
    pp = 25

if not force_cpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

from torch.utils.data import Dataset, DataLoader

class MinesweeperDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create a single sample
data = np.zeros((1, 10, ROWS, COLS), dtype=bool)
labels = np.zeros((1, ACTIONS), dtype=bool)

# Create the dataset
dataset = MinesweeperDataset(data, labels)
# Create the dataloader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

input_output_pairs = []

# Define the training loop
for episode in range(EPISODES):
    
    # exponential decay of epsilon
    EPSILON = EPSILON * 0.999
    EPSILON = max(EPSILON, 0.001)

    optimizer.zero_grad()

    score = 0
    done = False

    # take a random action if the board is empty
    first_move = True
    
    board, mine_board = reset_board()
    
    action = np.random.randint(0, ACTIONS)

    done = True

    while done:
        x = action // ROWS
        y = action % COLS
        
        if mine_board[x, y] == True:
            done = True
            action = np.random.randint(0, ACTIONS)
        else:
            done = False

    done = False
     
    state = board
    
    while not done:
        invalid_actions = np.nonzero(board[CLOSED].flatten() == False)[0]

        if np.random.rand() < EPSILON and not first_move:
            action = np.random.randint(0, ACTIONS)

            while action in invalid_actions:
                action = np.random.randint(0, ACTIONS)

        elif not first_move:
            action = model(torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device))

            action = action.detach().cpu().numpy()

            action[0, invalid_actions] = -np.inf

            action = np.argmax(action)

        if first_move:
            first_move = False

        # render every 100 episodes with pygame
        if episode % 1 == 0 and render_pygame:
            screen.fill((255, 255, 255))

            # undo one hot encoding and draw the board
            game_board = np.argmax(board, axis=0)


            for y in range(ROWS):
                for x in range(COLS):
                    if board[CLOSED, y, x] == True:
                        pygame.draw.rect(screen, (144, 238, 144), (pp + x * square_size, pp + y * square_size, square_size, square_size))
                    elif board[0, y, x] == True:
                        pygame.draw.rect(screen, (255, 255, 255), (pp + x * square_size, pp + y * square_size, square_size, square_size))
                    else:
                        pygame.draw.rect(screen, (255, 255, 255), (pp + x * square_size, pp + y * square_size, square_size, square_size))
                        text = str(game_board[y, x].item())
                        font = pygame.font.SysFont('Arial', 30)
                        text_surface = font.render(text, True, (0, 0, 0))
                        screen.blit(text_surface, (pp + x * square_size + 20, pp + y * square_size + 10))

                    if mine_board[y, x] == True:
                        pygame.draw.rect(screen, (255, 0, 0), (pp + x * square_size + 10, pp + y * square_size + 10, square_size - 20, square_size - 20))

                    if y == int(action // ROWS) and x == int(action % COLS):
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
                    pygame.draw.line(screen, (0, 0, 255), (pp + int(action_line[i] % ROWS) * square_size + 5, pp + int(action_line[i] // COLS) * square_size + 5), (pp + int(action_line[i + 1] % ROWS) * square_size + 5, pp + int(action_line[i + 1] // COLS) * square_size + 5))
            
            
            if one_hot_board:
                # draw the 10 small boards from one hot encoding 5 wide and 2 tall to the right of the big board start from the top left corner
                for i in range(10):
                    for x, y in itertools.product(range(ROWS), range(COLS)):
                        if board[i, x, y] == True:
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
                pygame.time.delay(200)
        
        
        # Take the action
        new_state, reward, done = step(board, action)

        model_state = model(torch.tensor(np.expand_dims(state, axis=0)).float().to(device))
        
        new_action = model(torch.tensor(np.expand_dims(new_state, axis=0)).float().to(device))
        
        new_action = new_action.detach().cpu().numpy()

        new_action[0, invalid_actions] = -np.inf

        new_action = np.argmax(new_action)

        # Update the state
        state = new_state
        
        score += reward
        
        if done:
            for i in range(ACTIONS):
                x = i // ROWS
                y = i % COLS

                new_board, reward, done = step(board, i)
                
                if reward == 100:
                    action_reward = 100
                elif mine_board[x, y] == True:
                    action_reward = reward
                else:
                    action_reward = reward
                
                done = True
                
                input_output_pairs.append((state, i, action_reward, done))

        else:
            input_output_pairs.append((state, action, reward, done))

    total_loss = 0

    for state, action, reward, done in input_output_pairs:
        # Get the reward for this action
        # Compute the loss for this input-output pair

        if done:
            loss = loss_fn(q_val, torch.tensor(reward).float().to(device))
        else:
            loss = loss_fn(q_val, torch.tensor(reward + GAMMA * new_action).float().to(device))
        # Accumulate the loss
        total_loss += loss

    # Compute the gradients
    avg_loss = total_loss / len(input_output_pairs)
    avg_loss.backward()
    # Update the model parameters
    optimizer.step()
    
    # Clear the list of input-output pairs
    input_output_pairs = []


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
    losses.append(loss.item())

    avg_score = np.mean(scores[-1000:])
    avg_loss = np.mean(losses[-1000:])

    if episode % 1 == 0:
        print(f'Episode: {episode:5}, Score: {score:5}, Avg Score: {avg_score:.2f}, Epsilon: {EPSILON:.4f}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}')

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

    



