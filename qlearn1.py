import sys
from six import StringIO
from random import randint

import numpy as np
import gym
from gym import spaces

# cell values, non-negatives indicate number of neighboring mines
MINE = -1
CLOSED = -2


def board2str(board, end='\n'):
    """
    Format a board as a string
    Parameters
    ----
    board : np.array
    end : str
    Returns
    ----
    s : str
    """
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            s += str(board[x][y]) + '\t'
        s += end
    return s[:-len(end)]


def is_new_move(my_board, x, y):
    """ return true if this is not an already clicked place"""
    return my_board[x, y] == CLOSED


def is_valid(board_size, x, y):
    """ returns if the coordinate is valid"""
    return (x >= 0) & (x < board_size) & (y >= 0) & (y < board_size)


def is_win(my_board, board_size, num_mines):
    """ return if the game is won """
    return np.count_nonzero(my_board == CLOSED) == num_mines


def is_mine(board, x, y):
    """return if the coordinate has a mine or not"""
    return board[x, y] == MINE


def place_mines(board_size, num_mines):
    """generate a board, place mines randomly"""
    mines_placed = 0
    board = np.zeros((board_size, board_size), dtype=int)
    while mines_placed < num_mines:
        rnd = randint(0, board_size * board_size)
        x = int(rnd / board_size)
        y = int(rnd % board_size)
        if is_valid(board_size, x, y):
            if not is_mine(board, x, y):
                board[x, y] = MINE
                mines_placed += 1
    return board


class MinesweeperEnv(gym.Env):
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, board_size, num_mines):
        """
        Create a minesweeper game.
        Parameters
        ----
        board_size: int     shape of the board
            - int: the same as (int, int)
        num_mines: int   num mines on board
        """

        self.board_size = board_size
        self.num_mines = num_mines
        self.board = place_mines(board_size, num_mines)
        self.my_board = np.ones((board_size, board_size), dtype=int) * CLOSED
        self.valid_actions = np.ones((self.board_size, self.board_size), dtype=bool)

        self.observation_space = spaces.Box(low=-2, high=9,
                                            shape=(self.board_size, self.board_size), dtype=int)
        self.action_space = spaces.MultiDiscrete([self.board_size, self.board_size])

    def count_neighbour_mines(self, x, y):
        """return number of mines in neighbour cells given an x-y coordinate
            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        neighbour_mines = 0
        for _x in range(x - 1, x + 2):
            for _y in range(y - 1, y + 2):
                if is_valid(self.board_size, _x, _y):
                    if is_mine(self.board, _x, _y):
                        neighbour_mines += 1
        return neighbour_mines

    def open_neighbour_cells(self, my_board, x, y):
        """return number of mines in neighbour cells given an x-y coordinate
            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        for _x in range(x-1, x+2):
            for _y in range(y-1, y+2):
                if is_valid(self.board_size, _x, _y):
                    if is_new_move(my_board, _x, _y):
                        my_board[_x, _y] = self.count_neighbour_mines(_x, _y)
                        if my_board[_x, _y] == 0:
                            my_board = self.open_neighbour_cells(my_board, _x, _y)
        return my_board

    def get_next_state(self, state, x, y):
        """
        Get the next state.
        Parameters
        ----
        state : (np.array)   visible board
        x : int    location
        y : int    location
        Returns
        ----
        next_state : (np.array)    next visible board
        game_over : (bool) true if game over
        """
        my_board = state
        game_over = False
        if is_mine(self.board, x, y):
            my_board[x, y] = MINE
            game_over = True
        else:
            my_board[x, y] = self.count_neighbour_mines(x, y)
            if my_board[x, y] == 0:
                my_board = self.open_neighbour_cells(my_board, x, y)
        self.my_board = my_board
        return my_board, game_over

    def reset(self):
        """
        Reset a new game episode. See gym.Env.reset()
        Returns
        ----
        next_state : (np.array, int)    next board
        """
        self.board = place_mines(self.board_size, self.num_mines)
        self.my_board = np.ones((self.board_size, self.board_size), dtype=int) * CLOSED
        self.valid_actions = np.ones((self.board_size, self.board_size), dtype=bool)
        return self.my_board

    def step(self, action):  # sourcery skip: raise-specific-error
        """
        See gym.Env.step().
        Parameters
        ----
        action : np.array    location
        Returns
        ----
        next_state : (np.array)    next board
        reward : float        the reward for action
        done : bool           whether the game end or not
        info : {}
        """
        state = self.my_board
        x = action // self.board_size
        y = action % self.board_size

        next_state, reward, done, info = self.next_step(state, x, y)
        self.my_board = next_state
        self.valid_actions = (next_state == CLOSED)
        info['valid_actions'] = (next_state == CLOSED)
        return next_state, reward, done, info

    def next_step(self, state, x, y):
        """
        Get the next observation, reward, done, and info.
        Parameters
        ----
        state : (np.array)    visible board
        x : int    location
        y : int    location
        Returns
        ----
        next_state : (np.array)    next visible board
        reward : float               the reward
        done : bool           whether the game end or not
        info : {}
        """
        my_board = state
        if not is_new_move(my_board, x, y):
            return my_board, -10, False, {}
        while True:
            state, game_over = self.get_next_state(my_board, x, y)
            if not game_over:
                if is_win(state, self.board_size, self.num_mines):
                    return state, 5000, True, {}
                else:
                    return state, 10, False, {}
            else:
                return state, -500, True, {}

    def render(self, mode='human'):
        """
        See gym.Env.render().
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = board2str(self.board)
        outfile.write(s)
        if mode != 'human':
            return outfile



# Use pygame to render the environment
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

final_reward = 0
game_over = False

wins = 0
losses = 0

BOARD_SIZE = 5
NUM_MINES = 5
NUM_EPISODES = 10000
rendering_enabled = False

env = MinesweeperEnv(board_size=BOARD_SIZE, num_mines=NUM_MINES)

# deep q learning with cuda

# Use one hot encoding for the state, as BOARD_SIZE * BOARD_SIZE * 10
# 10 is the number of possible states for a cell

import torch.nn.functional as F

# Define the neural network
class Net(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Net, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
    
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions
    
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=10000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        self.Q_eval = Net(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
            
        else:
            action = np.random.choice(self.action_space)
            
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=int)
        
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_memory[batch]
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

# if rendering is disabled
if not rendering_enabled:
    agent = Agent(gamma=0.90, epsilon=1.0, batch_size=64, n_actions=BOARD_SIZE * BOARD_SIZE, eps_end=0.01, input_dims=BOARD_SIZE * BOARD_SIZE * 10, lr=0.003)

    scores, eps_history = [], []

    for k in range(NUM_EPISODES):
        done = False
        observation_ = env.reset()
        observation = np.zeros((10, BOARD_SIZE, BOARD_SIZE))
        
        for i, j in enumerate([-2, 0, 1, 2, 3, 4, 5, 6, 7, 8]):
            #one hot encoding
            observation[i,:,:] = (observation_ == j)
        
        observation = observation.flatten()
        
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
            n_state, reward, done, _ = env.step(action)
            
            new_state = np.zeros((10, BOARD_SIZE, BOARD_SIZE))
            
            for i, j in enumerate([-2, 0, 1, 2, 3, 4, 5, 6, 7, 8]):
                #one hot encoding
                new_state[i,:,:] = (n_state == j)
            
            new_state = new_state.flatten()
            
            agent.store_transition(observation, action, reward, new_state, done)
            agent.learn()
            observation = new_state
            score += reward
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-100:])
        
        print(f'episode {k} score {score} average score {avg_score} epsilon {agent.epsilon}')
        
    # plot learning curve
    x = [i+1 for i in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()
    
     
    
    
    
