import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from M_simple import*
from M_pygame import*
from M_plot   import*

# Model parameters
GAMMA = 0.99
EPSILON_DECAY = 0.001
LEARNING_RATE = 0.003
epsilon = 1.0

EPISODES = 1_000_000
BATCH_SIZE = 100
USE_GPU = False
TRAINING = True
MODEL_SAVE_PATH = "Baba1.pth"

BUFFER_SIZE = 100

# Game parameters
ROWS = 10
COLS = 10
MINES = 20


if USE_GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Buffers for: state, action, new_action, reward, done
state_buffer = np.zeros((BUFFER_SIZE, 10, ROWS, COLS), dtype=bool)
action_buffer = np.zeros(BUFFER_SIZE, dtype=int)
new_state_buffer = np.zeros((BUFFER_SIZE, 10, ROWS, COLS), dtype=bool)
new_action_buffer = np.zeros(BUFFER_SIZE, dtype=int)
reward_buffer = np.zeros(BUFFER_SIZE, dtype=int)
done_buffer = np.zeros(BUFFER_SIZE, dtype=bool)

# Define the network CNN with a kernel size of 5
model = nn.Sequential(
    nn.Conv2d(10, 32, kernel_size=5, stride=1, padding=1),
    nn.Sigmoid(),
    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
    nn.Sigmoid(),
    nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
    nn.Sigmoid(), 
    nn.Flatten(),
    nn.Linear(2048, ROWS*COLS) 
).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Load model at MODEL_SAVE_PATH
try:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("Model found")
except FileNotFoundError:
    print("Model not found")


def model_action(state):
    invalid_actions = np.logical_not(state[CLOSED].flatten()).nonzero()[0]
                
    observation = model(torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device))
    observation = observation.detach().cpu().numpy().flatten()
    observation[invalid_actions] = -np.inf

    return np.argmax(observation)


def main():
    scores = []
    losses = []
    action_line = []
    step_count = 0
    screen = setup_screen(ROWS, COLS)


    print(device)

    #----------------------------------------Training----------------------------------------
    for episode in range(EPISODES):
        epsilon = max(epsilon - EPSILON_DECAY, 0.001)

        state, mine_board = create_board(ROWS, COLS, MINES)
        solved_board = solve_board(state)
        
        score = 0
        done = False
        is_first_move = True
        
        while not done:
            step_count += 1
        
            if np.random.rand() < epsilon:
                valid_actions = state[CLOSED].flatten().nonzero()[0]
                action = np.random.choice(valid_actions)   
            else:
                action = model_action(state)
            
            if render_pygame:
                game_loop(state, mine_board, action, screen, rows, cols)
            
            new_state, reward, done = step(state, MINES, mine_board, solved_board, action, is_first_move)
            new_action = model_action(new_state)
            is_first_move = False

            state = new_state
            score += reward
            
            # UPDATE BUFFERS
            # buffer_idx = step_count % buffer_size
            # state_buffer[buffer_idx] = state
            # action_buffer[buffer_idx] = action
            # new_state_buffer[buffer_idx] = new_state
            # new_action_buffer[buffer_idx] = new_action
            # reward_buffer[buffer_idx] = reward
            # done_buffer[buffer_idx] = done
                

        # Train the model
        if step_count > BUFFER_SIZE:
            batch_idx = np.random.choice(BUFFER_SIZE, size=BATCH_SIZE)
            
            out = model(torch.tensor(state_buffer[batch_idx]).float().to(device))
            q_val = out[np.arange(BATCH_SIZE), action_buffer[batch_idx]]
            out_next = model(torch.tensor(new_state_buffer[batch_idx]).float().to(device))
            q_val_next = out_next[np.arange(BATCH_SIZE), new_action_buffer[batch_idx]]

            target = torch.tensor(reward_buffer[batch_idx]).float().to(device) + GAMMA * q_val_next * (1 - torch.tensor(done_buffer[batch_idx]).float().to(device))
            l = loss_fn(q_val, target.float())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        scores.append(score)
        
        if step_count > BUFFER_SIZE:
            losses.append(l.item())
        
        if episode % 10 == 0:
            avg_score = np.mean(scores[-1000:])
            print(f"episode {episode:>5}, score {score:>6.1f}, avg score {avg_score:>6.1f}, eps {epsilon:.3f}, losses {np.mean(losses[-1000:]):.3f}")
            
            # plot_scores(scores, episode, score, avg_score, epsilon)

        if not TRAINING: break

    torch.save(model.state_dict(), MODEL_SAVE_PATH)