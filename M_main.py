import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from M_simple import Minesweeper
from M_pygame import*
from M_plot   import*

# Model parameters
GAMMA = 0.99
LEARNING_RATE = 0.003
EPSILON_DECAY = 0.999
epsilon = 1.0

EPISODES = 1_000_000
BATCH_SIZE = 100
BUFFER_SIZE = 10_000
MODEL_SAVE_PATH = "Baba1.pth"
USE_GPU = False
training = True


# Game parameters
ROWS = 5
COLS = 5
MINES = 3
REWARD_WIN = 100
REWARD_LOSS = -100
REWARD_STEP = -1
REWARDS = (REWARD_WIN, REWARD_LOSS, REWARD_STEP)


if USE_GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Buffers for: state, action, new_action, reward, done
state_buffer = np.zeros((BUFFER_SIZE, 10, ROWS, COLS), dtype=bool)
action_buffer = np.zeros(BUFFER_SIZE, dtype=int)
new_state_buffer = np.zeros((BUFFER_SIZE, 10, ROWS, COLS), dtype=bool)
invalid_actions_buffer = np.zeros((BUFFER_SIZE, ROWS*COLS), dtype=int)
reward_buffer = np.zeros(BUFFER_SIZE, dtype=int)
done_buffer = np.zeros(BUFFER_SIZE, dtype=bool)

# Define the network CNN with a kernel size of 5
model = nn.Sequential(
    nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
    nn.Sigmoid(),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.Sigmoid(),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.Sigmoid(), 
    nn.Flatten(),
    nn.Linear(3200, ROWS*COLS)
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
    invalid_actions = np.logical_not(state[-1].flatten()).nonzero()[0]
    
    with torch.inference_mode():
        observation = model(torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device))
        observation = observation.detach().cpu().numpy().flatten()
    observation[invalid_actions] = -np.inf

    return np.argmax(observation)


if __name__ == "__main__":
    scores = []
    # losses = []
    win_counter = 0
    loss_counter = 0
    action_line = []
    loss = 0
    step_count = 0
    screen = setup_screen(ROWS, COLS)

    print(device)

    for episode in range(EPISODES):
        epsilon = max(epsilon * EPSILON_DECAY, 0.001)

        env = Minesweeper((ROWS, COLS), MINES, REWARDS)
        state, mine_board, solved_board = env.board, env.mine_board, env.solved_board
        
        score = 0
        done = False
        first_move = True
        reward = 0
        
        while not done:
            step_count += 1
        
            if np.random.rand() < epsilon:
                valid_actions = state[-1].flatten().nonzero()[0]
                action = np.random.choice(valid_actions)
            else:
                action = model_action(state)
            
            game_loop(state, mine_board, action, screen, ROWS, COLS, reset=is_first_move)
            
            new_state, solved_board, reward, done = step(state, MINES, mine_board, solved_board, action, is_first_move)
            new_action = model_action(new_state)
            is_first_move = False

            
            invalid_actions_idx = np.logical_not(new_state[-1].flatten()).nonzero()[0]
            invalid_actions_new = np.zeros(ROWS*COLS, dtype=int)
            invalid_actions_new[invalid_actions_idx] = 1

            state = new_state
            score += reward
            
            # UPDATE BUFFERS
            buffer_idx = step_count % BUFFER_SIZE
            state_buffer[buffer_idx] = state
            action_buffer[buffer_idx] = action
            new_state_buffer[buffer_idx] = new_state
            invalid_actions_buffer[buffer_idx] = invalid_actions_new
            reward_buffer[buffer_idx] = reward
            done_buffer[buffer_idx] = done
        
        if reward == WIN_REWARD:
            win_counter += 1
        else:
            loss_counter += 1                

        # Train the model
        if step_count > BUFFER_SIZE:
            batch_idx = np.random.choice(BUFFER_SIZE, size=BATCH_SIZE)
        
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
        
        # if step_count > BUFFER_SIZE:
        #     losses.append(l.item())
        
        if episode % 10 == 0:
            # avg_score = np.mean(scores[-100:])
            print(f"episode {episode:>5}, score {score:>4}, eps {epsilon:.3f}, loss {1}, w/l {win_counter/loss_counter:>3.2f}")
            #, avg score {avg_score:>6.2f}
            # plot_scores(scores, episode, score, avg_score, epsilon)

        if not training: break

    torch.save(model.state_dict(), MODEL_SAVE_PATH)