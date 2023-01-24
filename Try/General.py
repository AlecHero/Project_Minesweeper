import itertools
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import pygame

# Pygame controls:
# Press R to render the game (learning is a lot faster without render)
# Press M to turn slow_mode off
# Press N to turn slow_mode on
# Press L to turn actionline on and off
# Press S to save the model where you are, and to show a graph of the average score over the last 1000 episodes

#-----------------------------------------Params--------------------------------------------------
# Define the parameters
GAMMA = 0.99
EPISODES = 10_000
EPSILON = 1
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01
BATCH_SIZE = 100
MEM_SIZE = 10_000
LEARNING_RATE = 0.0001
MODEL_NAME = "modelname.pth" # Name of the model to load or new model to save if not already trained 
# Ajust the kernel size to the the model if already trained
KERNEL_SIZE = 5
N1 = 32
N2 = 64
N3 = 128

# Define game parameters
ROWS = 9
COLS = 9
MINES = 10
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
running = True
training = True

if not force_cpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

#seed torch and numpy
torch.manual_seed(1)
np.random.seed(1)

#-----------------------------------------Pygame Params-------------------------------------------
square_size = 40
small_square_size = 15
slow_mode = False
render_pygame = True
action_line_enabled = False
one_hot_board = True
num_size = 25
pp = 25
action_line = []

pygame.init()

# one hot board
if one_hot_board:
    board_width = (ROWS) * square_size + (25 + small_square_size * ROWS) * 5 + pp
    board_height = (COLS) * square_size + 2 * pp + 250
else:
    board_width = (ROWS) * square_size + 2 * pp
    board_height = (COLS) * square_size + 2 * pp

screen = pygame.display.set_mode((board_width, board_height))


# -----------------------------------------Pygame--------------------------------------------------
def render():
    screen.fill((255, 255, 255))

    # undo one hot encoding and draw the board
    game_board = np.argmax(state, axis=0)


    for y, x in itertools.product(range(ROWS), range(COLS)):
        if state[CLOSED, y, x] == 1:
            pygame.draw.rect(screen, (144, 238, 144), (pp + x * square_size, pp + y * square_size, square_size, square_size))
        elif state[0, y, x] == 1:
            pygame.draw.rect(screen, (255, 255, 255), (pp + x * square_size, pp + y * square_size, square_size, square_size))
        else:
            pygame.draw.rect(screen, (255, 255, 255), (pp + x * square_size, pp + y * square_size, square_size, square_size))
            text = str(game_board[y, x].item())
            font = pygame.font.SysFont('Arial', num_size)
            font.set_bold(True)
            if text == "1":
                text_surface = font.render(text, True, (0, 0, 255))
            elif text == "2":
                text_surface = font.render(text, True, (0, 255, 0))
            elif text == "3":
                text_surface = font.render(text, True, (255, 0, 0))
            elif text == "4":
                text_surface = font.render(text, True, (0, 0, 128))
            elif text == "5":
                text_surface = font.render(text, True, (128, 0, 0))
            elif text == "6":
                text_surface = font.render(text, True, (0, 128, 128))
            elif text == "7":
                text_surface = font.render(text, True, (0, 0, 0))
            elif text == "8":
                text_surface = font.render(text, True, (128, 128, 128))
            screen.blit(text_surface, (pp + x * square_size + square_size / 2 - 5, pp + y * square_size + square_size / 2 - num_size / 2))
            font.set_bold(False)

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
    text_surface = font.render(f"Step: {str(step_count)}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 10))

    # show the episode number on the bottom of the screen
    text_surface = font.render(f"Episode: {str(episode)}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 30))

    # show the reward on the bottom of the screen
    text_surface = font.render(f"Reward: {str(reward)}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 50))

    # show the score
    text_surface = font.render(f"Score: {str(score)}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 70))

    #choose a font that has the same width for all characters
    text_surface = font.render(f"Action: {str(action)}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 90))

    with torch.inference_mode():
        outy = model(torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device))
        q_valsy = outy[0,action]
        out_nexty = model(torch.tensor(np.expand_dims(new_state, axis=0)).float().to(device))
        out_nexty[0, torch.tensor(invalid_actions_new).bool()] = -np.inf
        q_vals_nexty = out_nexty[0,torch.argmax(out_nexty)] 
        targety = reward + GAMMA * q_vals_nexty * (1 - done) 

    # show q_val
    text_surface = font.render(f"Qval: {str(round(q_valsy.item(), 2))}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 110))

    # show target
    text_surface = font.render(f"Target: {str(round(targety.item(), 2))}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 130))

    # USE MSE LOSS
    lossy = (q_valsy - targety) ** 2
    text_surface = font.render(f"Loss: {str(round(lossy.item(), 2))}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 150))

    # show the epsilon
    text_surface = font.render(f"Epsilon: {str(EPSILON)}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 170))

    # show done
    text_surface = font.render(f"Done: {str(done)}", True, (0, 0, 0))
    screen.blit(text_surface, (10, ROWS * square_size + pp + 190))

    if slow_mode:    
        pygame.time.delay(200)
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

    # Uncomment this if you don't want to open cells around 0 mines
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


# Step function
def step(board, action):
    done = False
    x = action // ROWS
    y = action % COLS

    if mine_board[x, y]:
        reward = LOSS_REWARD
        done = True

        return board, reward, done

    # Reward GUESS_REWARD each step, unless you open a cell with a number in the neighbours
    min_x = max(0, x-1)
    max_x = min(ROWS-1, x+1)
    min_y = max(0, y-1)
    max_y = min(COLS-1, y+1)
    reward = (
        GUESS_REWARD
        if board[CLOSED, min_x : max_x + 1, min_y : max_y + 1].all()
        else STEP_REWARD
    )
    
    # Recursively open the neighbours  
    board = open_neighbour_cells(board, mine_board, x, y)

    if np.sum(board[CLOSED]) == MINES:
        reward = WIN_REWARD
        done = True

    return board, reward, done

# Create a buffer for state, action, new_action, reward, done
buffer_size = MEM_SIZE
state_buffer = np.zeros((buffer_size, 10, ROWS, COLS), dtype=bool)
action_buffer = np.zeros(buffer_size, dtype=int)
new_state_buffer = np.zeros((buffer_size, 10, ROWS, COLS), dtype=bool)
invalid_actions_buffer = np.zeros((buffer_size, ACTIONS), dtype=bool)
reward_buffer = np.zeros(buffer_size, dtype=int)
done_buffer = np.zeros(buffer_size, dtype=bool)


# Define the model
model = nn.Sequential(
    nn.Conv2d(10, N1, kernel_size=KERNEL_SIZE, stride=1, padding="same"),
    nn.LeakyReLU(),
    nn.Conv2d(N1, N2, kernel_size=KERNEL_SIZE, stride=1, padding="same"),
    nn.LeakyReLU(),
    nn.Conv2d(N2, N3, kernel_size=KERNEL_SIZE, stride=1, padding="same"),
    nn.LeakyReLU(), 
    nn.Flatten(),
    nn.Linear(N3*ACTIONS, ACTIONS)
).to(device)

model.to(device)

# Print the model parameters
params = model.parameters()
parameters = [np.prod(p.size()) for p in params]
print("Model parameters:", parameters)
print("Total parameters:", sum(parameters))

print(device)
print(model)

# Load the model if it exists
try:
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
    print("Model loaded")
except FileNotFoundError:
    print("Model not found")

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define the loss function as mean absolute error
loss_fn = nn.MSELoss()

#----------------------------------------Training----------------------------------------
for episode in range(EPISODES):
    if step_count > buffer_size and training:
        EPSILON = EPSILON * EPSILON_DECAY
        EPSILON = max(EPSILON, EPSILON_MIN)
    
    score = 0
    done = False

    state, mine_board = reset_board()
    
    first_action = True
    
    while not done:
        invalid_actions = np.nonzero(state[CLOSED].flatten() == False)[0]

        if np.random.rand() < EPSILON and training:
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

        # Uncomment if you don't wan't to make sure the first action is not a mine
        if first_action:
            first_action = False
            while mine_board[action // ROWS, action % COLS]:
                state, mine_board = reset_board()
            
        
        new_state, reward, done = step(state, action)
        
        
        if reward == LOSS_REWARD:
            lost_counter += 1
        elif reward == WIN_REWARD:
            won_counter += 1
        
        # Invalid actions
        invalid_actions_idx = np.nonzero(new_state[CLOSED].flatten() == False)[0]
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
          
        # Pygame render
        if episode % 1 == 0 and render_pygame:
            render()
            
        state = new_state
        
        # Pygame controls:
        # Press R to render the game (learning is a lot faster without render)
        # Press M to turn slow_mode off
        # Press N to turn slow_mode on
        # Press L to turn actionline on and off
        # Press S to save the model where you are, and to show a graph of the average score over the last 1000 episodes
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                slow_mode = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                slow_mode = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                render_pygame = not render_pygame
            if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                action_line_enabled = not action_line_enabled
        
    # Runs at the end of each episode
    if step_count > buffer_size and training:
        batch_idx = np.random.choice(buffer_size, size=BATCH_SIZE)
        
        # Find the q values for the actions taken
        out = model(torch.tensor(state_buffer[batch_idx]).float().to(device))
        q_vals = out[np.arange(BATCH_SIZE), action_buffer[batch_idx]]
        
        # Find the q values for the next state
        out_next = model(torch.tensor(new_state_buffer[batch_idx]).float().to(device))
        
        # Remove invalid actions
        out_next[torch.tensor(invalid_actions_buffer[batch_idx]).bool()] = -np.inf
        q_vals_next = out_next[np.arange(BATCH_SIZE), torch.argmax(out_next, dim=1)] 
        
        # Calculate the target
        reward_tensor = torch.tensor(reward_buffer[batch_idx]).float().to(device)
        done_tensor = torch.tensor(done_buffer[batch_idx]).float().to(device)
        with torch.no_grad():
            target = reward_tensor + GAMMA * q_vals_next * (1 - done_tensor) 
        
        # Calculate the loss
        l = loss_fn(q_vals, target)

        # Backpropagation
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    scores.append(score)
    
    if step_count > buffer_size and training:
        losses.append(l.item())
    
    lossy = 0
    avg_score = 0
    avg_loss = 0
    
    # Change 1 to a higher number to print less often
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
    if not running:
        break



#----------------------------------------Save model-------------------------------------------------
if training:
    torch.save(model.state_dict(), MODEL_NAME)

    # Save scores and losses to file
    with open(f"{MODEL_NAME}_scores.txt", "w") as f:
        for item in scores:
            f.write(f"{item},")
            
    with open(f"{MODEL_NAME}_losses.txt", "w") as f:
        for item in losses:
            f.write(f"{item},")

#----------------------------------------Plot learning curve----------------------------------------
# plot learning curve
x = [i+1 for i in range(len(scores))]
running_avg = np.zeros(len(scores))
for i in range(len(running_avg)):
    running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    
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
