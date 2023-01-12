import itertools
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import pygame

#-----------------------------------------Params--------------------------------------------------
# Define the parameters
GAMMA = 0.99
EPISODES = 1_000_000
EPSILON = 1
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01
BATCH_SIZE = 50
MEM_SIZE = 10_000
LEARNING_RATE = 0.0001
MODEL_NAME = "6x64mMom.pth"

# Define game parameters
ROWS = 6
COLS = 6
MINES = 4
CLOSED = 9
LOSS_REWARD = -100
WIN_REWARD = 100
STEP_REWARD = 30
GUESS_REWARD = -30

# Counters
scores = []
losses = []
step_count = 0
lost_counter = 0
won_counter = 0

# Define the actions
ACTIONS = ROWS * COLS

# Pytorch params
force_cpu = False
training = True

if not force_cpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")



#-----------------------------------------Pygame Params-------------------------------------------
square_size = 40
small_square_size = 15
slow_mode = False
render_pygame = False
action_line_enabled = False
one_hot_board = True
num_size = 25
pp = 25
action_line = []

pygame.init()

# make space for 10 small boards on the right side of the big board in the center
if one_hot_board:
    board_width = (ROWS) * square_size + (25 + small_square_size * ROWS) * 5 + pp
    board_height = (COLS) * square_size + 2 * pp + 250
else:
    board_width = (ROWS) * square_size
    board_height = (COLS) * square_size

screen = pygame.display.set_mode((board_width, board_height))


# -----------------------------------------Pygame--------------------------------------------------
def render():
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
                font = pygame.font.SysFont('Arial', num_size)
                text_surface = font.render(text, True, (0, 0, 0))
                screen.blit(text_surface, (pp + x * square_size + square_size / 2, pp + y * square_size + square_size / 2 - num_size / 2))

            if mine_board[y, x] == True:
                pygame.draw.rect(screen, (255, 0, 0), (pp + x * square_size + square_size / 4, pp + y * square_size + square_size / 4, square_size / 2, square_size / 2))

            if y == int(action // ROWS) and x == int(action % COLS):
                pygame.draw.rect(screen, (0, 0, 255), (pp + x * square_size + 5, pp + y * square_size + 5, square_size / 4, square_size / 4))


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
                    pygame.draw.rect(screen, (102, 153, 204), (ROWS * square_size + 2 * pp + (i % 5) * (ROWS + 1) * small_square_size + x * small_square_size, pp + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size, small_square_size, small_square_size))
                else:
                    pygame.draw.rect(screen, (255, 255, 255), (ROWS * square_size + 2 * pp + (i % 5) * (ROWS + 1) * small_square_size + x * small_square_size, pp + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size, small_square_size, small_square_size))

            # render the grid
            for x in range(ROWS + 1):
                pygame.draw.line(screen, (0, 0, 0), (ROWS * square_size + 2 * pp + (i % 5) * (ROWS + 1) * small_square_size + x * small_square_size, pp + (i // 5) * (COLS + 1) * small_square_size), (ROWS * square_size + 2 * pp + (i % 5) * (ROWS + 1) * small_square_size + x * small_square_size, pp + (i // 5) * (COLS + 1) * small_square_size + COLS * small_square_size))
            for y in range(COLS + 1):
                pygame.draw.line(screen, (0, 0, 0), (ROWS * square_size + 2 * pp + (i % 5) * (ROWS + 1) * small_square_size, pp + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size), (ROWS * square_size + 2 * pp + (i % 5) * (ROWS + 1) * small_square_size + ROWS * small_square_size, pp + (i // 5) * (COLS + 1) * small_square_size + y * small_square_size))     
            
    font = pygame.font.SysFont('Arial', 20)
    # show the episode number on the bottom of the screen
    text_surface = font.render("Step: " + str(step_count), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 10))
    
    # show the episode number on the bottom of the screen
    text_surface = font.render("Episode: " + str(episode), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 30))
    
    # show the reward on the bottom of the screen
    text_surface = font.render("Reward: " + str(reward), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 50))

    # show the score
    text_surface = font.render("Score: " + str(score), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 70))
    
    #choose a font that has the same width for all characters
    text_surface = font.render("Action: " + str(action), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 90))
    
    with torch.inference_mode():
        outy = model(torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device))
        q_valsy = outy[0,action]
        out_nexty = model(torch.tensor(np.expand_dims(new_state, axis=0)).float().to(device))
        out_nexty[0, torch.tensor(invalid_actions_new).bool()] = -np.inf
        q_vals_nexty = out_nexty[0,torch.argmax(out_nexty)] 
        targety = reward + GAMMA * q_vals_nexty * (1 - done) 
    
    # show q_val
    text_surface = font.render("Qval:  " + str(round(q_valsy.item(), 2)), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 110))
    
    # show target
    text_surface = font.render("Target: " + str(round(targety.item(), 2)), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 130))
    
    # USE MSE LOSS
    lossy = (q_valsy - targety) ** 2
    text_surface = font.render("Loss: " + str(round(lossy.item(), 2)), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 150))
    
    # show the epsilon
    text_surface = font.render("Epsilon: " + str(EPSILON), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 170))
    
    # show done
    text_surface = font.render("Done: " + str(done), True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 190))

    if slow_mode:    
        pygame.time.delay(500)
    pygame.display.update()


# ----------------------------------------Minesweeper game----------------------------------------
# place mines
def place_mines(mine_board):
    mines = np.random.choice(ROWS*COLS, MINES, replace=False)

    for mine in mines:
        x = mine // ROWS
        y = mine % COLS
        mine_board[x, y] = True

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

    if mines == 0:
        for _x in range(max(0, x-1), min(ROWS, x+2)):
            for _y in range(max(0, y-1), min(COLS, y+2)):
                if new_board[CLOSED, _x, _y]:
                    mines = count_neighbour_mines(mine_board, _x, _y)
                    if mines == 0:
                        new_board = open_neighbour_cells(new_board, mine_board, _x, _y)
                    else:
                        new_board[mines, _x, _y] = True
                        new_board[CLOSED, _x, _y] = False

    return new_board


# def step
def step(board, action):
    done = False
    x = action // ROWS
    y = action % COLS
    
    if mine_board[x, y]:
        reward = LOSS_REWARD
        done = True
        
        return board, reward, done
    
    min_x = max(0, x-1)
    max_x = min(ROWS-1, x+1)
    min_y = max(0, y-1)
    max_y = min(COLS-1, y+1)
    if not board[CLOSED, min_x:max_x+1, min_y:max_y+1].all():
        reward = STEP_REWARD
    else:
        reward = GUESS_REWARD
        
    board = open_neighbour_cells(board, mine_board, x, y)
    
    if np.sum(board[CLOSED]) == MINES:
        reward = WIN_REWARD
        done = True
        
    return board, reward, done

# create a buffer for state, action, new_action, reward, done
buffer_size = MEM_SIZE
state_buffer = np.zeros((buffer_size, 10, ROWS, COLS), dtype=bool)
action_buffer = np.zeros(buffer_size, dtype=int)
new_state_buffer = np.zeros((buffer_size, 10, ROWS, COLS), dtype=bool)
invalid_actions_buffer = np.zeros((buffer_size, ACTIONS), dtype=bool)
reward_buffer = np.zeros(buffer_size, dtype=int)
done_buffer = np.zeros(buffer_size, dtype=bool)

# Define the network CNN with a kernel size of 5
model = nn.Sequential(
    nn.Conv2d(10, 32, kernel_size=5, stride=1, padding=2),
    nn.LeakyReLU(),
    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
    nn.LeakyReLU(),
    nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
    nn.LeakyReLU(), 
    nn.Flatten(),
    nn.Linear(4608, ACTIONS)
).to(device)

model.to(device)

print(device)
print(model)

# load the model if it exists
try:
    model.load_state_dict(torch.load(MODEL_NAME))
    print("Model loaded")
except FileNotFoundError:
    print("Model not found")

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Define the loss function as mean absolute error
loss_fn = nn.MSELoss()


#----------------------------------------Training----------------------------------------
for episode in range(EPISODES):
    EPSILON = EPSILON * EPSILON_DECAY
    EPSILON = max(EPSILON, EPSILON_MIN)
    
    score = 0
    done = False

    state, mine_board = reset_board()
    
    first_action = True
    
    while not done:
        invalid_actions = np.nonzero(state[CLOSED].flatten() == False)[0]

        if np.random.rand() < EPSILON:
            action = np.random.randint(0, ACTIONS)
            
            while action in invalid_actions:
                action = np.random.randint(0, ACTIONS)    
        
        else:
            with torch.inference_mode():
                observation = model(torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device))
                observation = observation.detach().cpu().numpy()
                    
            observation = observation.flatten()
            
            observation[invalid_actions] = -np.inf
            
            action = np.argmax(observation)

        
        if first_action:
            first_action = False
            while mine_board[action // ROWS, action % COLS]:
                state, mine_board = reset_board()
        
        
        new_state, reward, done = step(state, action)
        
        
        if reward == LOSS_REWARD:
            lost_counter += 1
        elif reward == WIN_REWARD:
            won_counter += 1
        
        if episode % 1 == 0 and render_pygame:
            render()
        
        # invalid actions
        invalid_actions_idx = np.nonzero(new_state[CLOSED].flatten() == False, )[0]
        invalid_actions_new = np.zeros(ROWS * COLS)
        invalid_actions_new[invalid_actions_idx] = True
        
        
        score += reward
        
        
        buffer_idx = step_count % buffer_size
        state_buffer[buffer_idx] = state
        action_buffer[buffer_idx] = action
        new_state_buffer[buffer_idx] = new_state
        reward_buffer[buffer_idx] = reward
        done_buffer[buffer_idx] = done
        invalid_actions_buffer[buffer_idx] = invalid_actions_new 
        step_count += 1
          
            
        state = new_state
        
        
        # Pygame event handling
        # Tryk R for at render spillet (learning er en del hurtigere uden render)
        # Tryk M for at slå slow_mode fra
        # Tryk N for at slå slow_mode til
        # Tryk L for at slå actionline til og fra
        # Tryk S for at gemme modellen der hvor man er
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
        

    if step_count > buffer_size:
        batch_idx = np.random.choice(buffer_size, size=BATCH_SIZE)
        
        out = model(torch.tensor(state_buffer[batch_idx]).float().to(device))
        q_vals = out[np.arange(BATCH_SIZE), action_buffer[batch_idx]]
        
        out_next = model(torch.tensor(new_state_buffer[batch_idx]).float().to(device))
        out_next[torch.tensor(invalid_actions_buffer[batch_idx]).bool()] = -np.inf
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
    
    lossy = 0
    avg_score = 0
    avg_loss = 0
    
    if episode % 1 == 0:
        if len(losses) > 0:
            avg_score = round(np.mean(scores[-100:]), 1)
            avg_loss = round(np.mean(losses[-100:]), 0)
            lossy = round(losses[-1:][0], 0)
            score = round(score, 0)
        print(f'Episode {episode}, Step {step_count}, score {score:5}, avg_score {avg_score:7}, eps {EPSILON:.4f}, loss {lossy:7}, avg_loss {avg_loss:6}, won {won_counter}, lost {lost_counter}')
        
    
    # Pygame action line reset    
    if action_line_enabled and render_pygame:
        action_line = []
        
    # Break if stopped by pygame event
    if not training:
        break



#----------------------------------------Save model----------------------------------------
torch.save(model.state_dict(), MODEL_NAME)


#----------------------------------------Plot learning curve----------------------------------------
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
