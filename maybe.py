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
    
    # generate mines number of random unique numbers between 0 and 99
    mines = np.random.choice(ROWS*COLS, MINES, replace=False)
    
    for mine in mines:
        x = mine // ROWS
        y = mine % COLS
        mine_board[x, y] = True
         
    # Uncomment to the same board for each game
    # mine_board1 = np.array([[False, False, False, False, True, True, False, False, False, False],
    #                         [False, False, False, False, False, False, False, False, True, False],
    #                         [False, True, False, False, False, True, True, False, True, False],
    #                         [False, False, False, False, True, False, False, False, True, False],
    #                         [False, True, False, False, False, False, False, False, True, False],
    #                         [False, False, False, False, False, False, False, False, False, False],
    #                         [False, False, False, True, False, False, True, True, False, False],
    #                         [False, False, False, True, False, False, False, False, False, False],
    #                         [False, False, True, False, False, False, True, False, False, False],
    #                         [False, False, False, False, False, False, True, True, True, False]
    #                         ])
    
    # mine_board2 = np.array([[False, False, False, False, False, False, False, False, False, False],
    #                         [False, True, False, False, False, False, False, False, True, False],
    #                         [False, False, True, False, False, True, False, True, False, False],
    #                         [True, False, True, True, False, False, False, False, True, False],
    #                         [False, False, True, False, False, False, False, False, False, True],
    #                         [False, True, False, True, False, False, False, False, False, False],
    #                         [False, False, False, False, False, True, False, False, True, False],
    #                         [False, False, False, False, True, False, False, True, False, False],
    #                         [False, False, False, False, False, False, True, False, False, False],
    #                         [False, False, False, True, True, False, False, False, False, False]
    #                         ])

    # # choose a random board
    # if  np.random.randint(0, 2) == 0:
    #     mine_board = mine_board1
    # else:
    #     mine_board = mine_board2

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
    new_board = board.copy()
    new_board[CLOSED, x, y] = False

    mines = count_neighbour_mines(mine_board, x, y)
    new_board[mines, x, y] = True

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

    return new_board


def step(board, action):
    x = action // ROWS
    y = action % COLS
    
    if mine_board[x, y]:
        reward = -100
        done = True
        
    else:
        board = open_neighbour_cells(board, mine_board, x, y).copy()
        reward = 20
        done = False

    if np.sum(board[CLOSED]) == MINES:
        reward = 100
        done = True

    return board, reward, done

# Define the parameters
GAMMA = 0.99
EPISODES = 1_000_000
EPSILON = 1

# Define game parameters
ROWS = 10
COLS = 10
MINES = 20
BATCH_SIZE = 100

# Define the states
CLOSED = 9

# Define the actions
ACTIONS = ROWS * COLS

# ----------------------------------------Parameters----------------------------------------
force_cpu = False
training = True

square_size = 50
small_square_size = 20
slow_mode = False
render_pygame = True
action_line_enabled = False
one_hot_board = True
pp = 0

if not force_cpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

from torch.utils.data import Dataset, DataLoader

# create a buffer for state, action, new_action, reward, done
buffer_size = 10_000
state_buffer = np.zeros((buffer_size, 10, ROWS, COLS), dtype=bool)
action_buffer = np.zeros(buffer_size, dtype=int)
new_state_buffer = np.zeros((buffer_size, 10, ROWS, COLS), dtype=bool)
invalid_actions_buffer = np.zeros((buffer_size, ACTIONS), dtype=bool)
reward_buffer = np.zeros(buffer_size, dtype=int)
done_buffer = np.zeros(buffer_size, dtype=bool)

# Define the network CNN with a kernel size of 5
model = nn.Sequential(
    nn.Conv2d(10, 32, kernel_size=5, stride=1, padding=1),
    nn.Sigmoid(),
    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
    nn.Sigmoid(),
    nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
    nn.Sigmoid(), 
    nn.Flatten(),
    nn.Linear(2048, ACTIONS)
).to(device)

model.to(device)

print(device)
print(model)

# load the model if it exists
try:
    model.load_state_dict(torch.load("Baba3.pth"))
    print("Model loaded")
except FileNotFoundError:
    print("Model not found")

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Define the loss function as mean absolute error
loss_fn = nn.MSELoss()
scores = []
losses = []
action_line = []
step_count = 0

pygame.init()

# make space for 10 small boards on the right side of the big board in the center
if one_hot_board:
    board_width = (ROWS) * square_size * 3.5 - pp
    board_height = (COLS) * square_size + 2 * pp
else:
    board_width = (ROWS) * square_size
    board_height = (COLS) * square_size

screen = pygame.display.set_mode((board_width, board_height))


done_counter = 0
lost_counter = 0


#----------------------------------------Training----------------------------------------
for episode in range(EPISODES):
    EPSILON = EPSILON * 0.999
    EPSILON = max(EPSILON, 0.01)

    score = 0
    done = False

    first_move = True
    
    state, mine_board = reset_board()
    
    action = np.random.randint(0, ACTIONS)

    first_move_done = False

    while not first_move_done:
        x = action // ROWS
        y = action % COLS
        
        if mine_board[x, y] == True:
            first_move_done = False
            action = np.random.randint(0, ACTIONS)
        else:
            first_move_done = True
    
    while not done:
        invalid_actions = np.nonzero(state[CLOSED].flatten() == False)[0]

        if np.random.rand() < EPSILON and not first_move:
            action = np.random.randint(0, ACTIONS)

            while action in invalid_actions:
                action = np.random.randint(0, ACTIONS)
                
        elif not first_move:
            
            with torch.inference_mode():
                observation = model(torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device))
                observation = observation.detach().cpu().numpy()
                    
            observation = observation.flatten()
            
            observation[invalid_actions] = -np.inf
            
            action = np.argmax(observation)

        if first_move:
            first_move = False
        
        if episode % 1 == 0 and render_pygame:
            screen.fill((255, 255, 255))

            # undo one hot encoding and draw the board
            game_board = np.argmax(state, axis=0)


            for y in range(ROWS):
                for x in range(COLS):
                    if state[CLOSED, y, x] == 1:
                        pygame.draw.rect(screen, (144, 238, 144), (pp + x * square_size, pp + y * square_size, square_size, square_size))
                    elif state[0, y, x] == 1:
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
                        if state[i, y, x] == 1:
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
        
        new_state, reward, done = step(state, action)
        
        # illegal actions
        invalid_actions_idx = np.nonzero(new_state[CLOSED].flatten() == False, )[0]
        invalid_actions_new = np.zeros(ROWS * COLS)
        invalid_actions_new[invalid_actions_idx] = True
        
        score += reward
        
        if done:
            valid_actions = np.nonzero(new_state[CLOSED].flatten() == True)[0]
            invalid_actions_newy = np.zeros(ROWS * COLS)
            invalid_actions_newy[valid_actions] = False
            
            for i in valid_actions:
                x = i // ROWS
                y = i % COLS

                new_statey, rewardy, doney = step(state.copy(), i)
                
                
                # buffer to buffer at ind % buffer_size
                buffer_idx = step_count % buffer_size
                state_buffer[buffer_idx] = state
                action_buffer[buffer_idx] = i
                new_state_buffer[buffer_idx] = new_statey
                reward_buffer[buffer_idx] = rewardy
                done_buffer[buffer_idx] = doney
                invalid_actions_buffer[buffer_idx] = invalid_actions_newy
                step_count += 1
                
        else:
            buffer_idx = step_count % buffer_size
            state_buffer[buffer_idx] = state
            action_buffer[buffer_idx] = action
            new_state_buffer[buffer_idx] = new_state
            reward_buffer[buffer_idx] = reward
            done_buffer[buffer_idx] = done
            invalid_actions_buffer[buffer_idx] = invalid_actions_new 
            step_count += 1
            
        state = new_state

    # Train the model
    if step_count > buffer_size:
        batch_idx = np.random.choice(buffer_size, size=BATCH_SIZE)
        
        out = model(torch.tensor(state_buffer[batch_idx]).float().to(device))
        q_vals = out[np.arange(BATCH_SIZE), action_buffer[batch_idx]]
        out_next = model(torch.tensor(new_state_buffer[batch_idx]).float().to(device))
        
        out_next[torch.tensor(invalid_actions_buffer[batch_idx]).bool()] = -torch.inf
        
        q_vals_next = out_next[np.arange(BATCH_SIZE), torch.argmax(out_next, dim=1)] 
        
        reward_tensor = torch.tensor(reward_buffer[batch_idx]).float().to(device)
        done_tensor = torch.tensor(done_buffer[batch_idx]).float().to(device)
        
        with torch.no_grad():
            target = reward_tensor + GAMMA * q_vals_next * (1 - done_tensor) 
        l = loss_fn(q_vals, target)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    scores.append(score)
    
    if step_count > buffer_size:
        losses.append(l.item())
    
    if episode % 1 == 0:
        avg_score = np.mean(scores[-100:])
        print(f"Episode {episode}, Step {step_count}, score {score:4}, avg score {avg_score:4}, eps {EPSILON:.6f}, losses {np.mean(losses[-100:]):.4f}")
        
        
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

torch.save(model.state_dict(), 'Baba3.pth')
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
plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
