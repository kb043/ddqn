import math, random
import copy
import gym
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
from wrappers import make_atari, wrap_deepmind, wrap_pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if torch.cuda.is_available() else autograd.Variable(*args, **kwargs)

def update_target(current_model, target_model, tau):
    for param, target_param in zip(current_model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values0 = target_models[0](next_state)
    next_q_state_values1 = target_models[1](next_state)
    next_q_state_values2 = target_models[2](next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value0 = next_q_state_values0.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value0 * (1 - done)
    next_q_value1 = next_q_state_values1.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value1 = reward + gamma * next_q_value1 * (1 - done)
    next_q_value2 = next_q_state_values2.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value2 = reward + gamma * next_q_value2 * (1 - done)
    a = expected_q_value2 - 2 * expected_q_value1 + expected_q_value
    for i in range(batch_size):
        if a[i] == 0:
            expected_q_value[i] = expected_q_value[i]
        else:
            expected_q_value[i] = expected_q_value[i] - (expected_q_value1[i] - expected_q_value[i]).pow(2) / a[i]
    expected_q_value[expected_q_value == float('inf')] = 0
    expected_q_value[expected_q_value == float('-inf')] = 0

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def eval_policy(current_model, env_id, seed, eval_episodes=10):

    env = gym.make(env_id)
    env.seed(seed + 100)
    avg_reward = 0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = current_model.act(state, 0)
            state, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(state).unsqueeze(0))
                q_value = self.forward(state)
                action = q_value.max(1)[1].data[0].cpu().numpy()
        else:
            action = random.randrange(env.action_space.n)
        return action


env_id = "CartPole-v1"
env = gym.make(env_id)
seed = 0

env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

file_name = f"{'ddqn'}_{env_id}_{seed}"
print("---------------------------------------")
print(f"Env: {env_id}, Seed: {seed}")
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")

current_model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_models = []
for i in range(3):
    target_models.append(copy.deepcopy(target_model))

optimizer = optim.Adam(current_model.parameters())


replay_buffer = ReplayBuffer(5000)

update_target(current_model, target_model, 1)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

frame_idx = 0
batch_size = 32
gamma = 0.99
episodes = 500
evaluations = [eval_policy(current_model, env_id, seed)]

losses = []
all_rewards = []
episode_reward = 0
episode_num = 0
episode_timesteps = 0

state = env.reset()
done = False
while (episode_num < episodes):
    if (not done):
        episode_timesteps += 1
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        frame_idx += 1
        if len(replay_buffer) > batch_size:
            compute_td_loss(batch_size)
        if frame_idx % 2 == 0:
            update_target(current_model, target_model, 0.02)
            target_models.append(copy.deepcopy(target_model))
            del target_models[0]

    if done:
        state = env.reset()
        done = False
        all_rewards.append(episode_reward)
        print(
            f"Total T: {frame_idx} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        episode_reward = 0
        episode_num += 1
        episode_timesteps = 0
        np.save(f"./results/{file_name}", all_rewards)




