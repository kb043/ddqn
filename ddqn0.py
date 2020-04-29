import math, random

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

#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if torch.cuda.is_available() else autograd.Variable(*args, **kwargs)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def save(filename):
    torch.save(current_model.state_dict(), filename + "net")
    torch.save(optimizer.state_dict(), filename + "optimizer")

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def eval_policy(current_model, env_id, seed, eval_episodes=10):
    env = make_atari(env_id)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False)

    env.seed(seed + 100)
    avg_reward = 0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = current_model.act(state, 0.05)
            state, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

class LinearSchedule(object):
    def __init__(self, schedule_timesteps=100000, final_p=0.1, initial_p=1.0):

        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
                q_value = self.forward(state)
                action = q_value.max(1)[1].data[0].cpu()
        else:
            action = random.randrange(env.action_space.n)
        return action


env_id = "SpaceInvadersNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
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

if not os.path.exists("./models"):
    os.makedirs("./models")

current_model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
target_model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)

optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

replay_initial = 10000
replay_buffer = ReplayBuffer(1000000)

update_target(current_model, target_model)

epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

num_frames = 20000000
batch_size = 32
gamma = 0.99

evaluations = [eval_policy(current_model, env_id, seed)]

losses = []
all_rewards = []
episode_reward = 0
episode_num = 0
episode_timesteps = 0

state = env.reset()
exploration = LinearSchedule()
t = 0
for frame_idx in range(1, num_frames + 1):
    episode_timesteps += 1
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, 0.1)

    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        print(f"Total T: {frame_idx} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        episode_reward = 0
        episode_num += 1
        episode_timesteps = 0


    if len(replay_buffer) > replay_initial:
        compute_td_loss(batch_size)

    if frame_idx % 10000 == 0:
        update_target(current_model, target_model)

    if frame_idx % 10000 == 0:
        evaluations.append(eval_policy(current_model, env_id, seed))
        np.save(f"./results/{file_name}", evaluations)
        save(f"./models/{file_name}")


