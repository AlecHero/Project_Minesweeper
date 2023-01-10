import torch
import numpy as np
import matplotlib.pyplot as plt
import gym


def drawnow():
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()
plt.ion()


# Learning parameters
n_games = 1000
learning_rate = 0.005
gamma = 0.99
epsilon = 1.0
epsilon_step = 0.00001
epsilon_min = 0.01
batch_size = 64
buffer_size = 20000
step_count = 0


q_net = torch.nn.Sequential(
    torch.nn.Linear(8, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 4),
)
optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
loss = torch.nn.MSELoss()


env = gym.make("LunarLander-v2")
actions = np.arange(4)

obs_buffer = np.zeros((buffer_size, 8))
obs_next_buffer = np.zeros((buffer_size, 8))
action_buffer = np.zeros(buffer_size)
reward_buffer = np.zeros(buffer_size)
done_buffer = np.zeros(buffer_size)


for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        
        epsilon = np.maximum(epsilon-epsilon_step, epsilon_min)
        if np.random.rand() < epsilon:
            observation = np.random.choice(actions)
        else:
            action = np.argmax(q_net(torch.tensor(observation)).detach().numpy())
        observation_next, reward, done, _ = env.step(action)
        
        score += reward
        
        buf_idx = step_count % buffer_size
        obs_buffer[buf_idx] = observation
        obs_next_buffer[buf_idx] = observation_next
        action_buffer[buf_idx] = action
        reward_buffer[buf_idx] = reward
        done_buffer[buf_idx] = done
        
        observation = observation_next

        if step_count > buffer_size:
            batch_idx = np.random.choice(buffer_size, size=batch_size)
            
            out = q_net(torch.tensor(obs_buffer[batch_idx].float()))
            q_val = out[np.arange(batch_size), action_buffer[batch_idx]]
            out_next = q_net(torch.tensor(obs_next_buffer[batch_idx].float()))
            with torch.no_grad():
                target = torch.tensor(reward_buffer[batch_idx].float()) + gamma * torch.max(out_next, dim=1).values*(1-done_buffer[batch_idx])
            l = loss(q_val, target)
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    
    scores.append(score)
    print(f"Score = {score}")
    
    plt.clf()
    plt.plot(scores)
    plt.title(f"Step {step_count}, eps={epsilon:.3f}")
    plt.ylim(-500,300)
    drawnow()
        

env = gym.make("LunarLander-v2")
observation = env.reset()
done = False
while not done:
    action = torch.argmax(q_net(torch.tensor(observation)))
    observation, reward, done, _, = env.step(action)
    env.render()
