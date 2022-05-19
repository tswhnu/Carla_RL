
####new
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# the defination of the hyper parameters
BATCH_SIZE = 16  # batch size of the training data
LR = 0.001  # learning rate
EPSILON = 0.6  # greedy algorithm
GAMMA = 0.9  # reward discount
TARGET_UPDATE = 100  # update the target network after training
MEMORY_CAPACITY = 4000  # the capacity of the memory
N_STATE = 4  # the number of states that can be observed from environment
ACTION_DIM = 1  # the number of action that the agent should have
N_CHANNEL = 6
# decide the device used to train the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the structure of the network

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000


# input: st, output:at
class Actor(nn.Module):
    def __init__(self, n_state=N_STATE, action_dim=ACTION_DIM):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_state, 64)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc2 = nn.Linear(64,action_dim)
        self.fc2.weight.data.normal_(0, 0.01)

    def forward(self, s):
        x = s.to(device)
        x = F.relu(self.fc1(x))
        out = torch.tanh(self.fc2(x))
        return out


# input: st, actor(st) output: q-value
class Critic(nn.Module):
    def __init__(self, n_state=N_STATE):
        super().__init__()
        self.fc1 = nn.Linear(n_state + 1, 64)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc2 = nn.Linear(64, 1)
        self.fc2.weight.data.normal_(0, 0.01)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DDPG(object):
    def __init__(self, **kwargs):

        self.tau = 0.02
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        self.actor, self.actor_target = Actor().to(device), Actor().to(device)
        self.critic, self.critic_target = Critic().to(device), Critic().to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.memory_counter = 0
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().detach().numpy()[0]
        action += random.gauss(0, 0.1)
        if action >= 1.0:
            action = 1.0
        elif action <= -1.0:
            action = -1.0
        return action

    def store_transition(self, s, a, r, s_, done):
        transition = [s, a, r, s_, done]
        self.memory.append(transition)
        self.memory_counter += 1

    def optimize_model(self):
        # get the samples to train the policy net
        sample_batch = random.sample(self.memory, BATCH_SIZE)
        batch_s = torch.FloatTensor(np.array([transition[0] for transition in sample_batch])).to(device)
        batch_a = torch.LongTensor(np.array([transition[1] for transition in sample_batch])).unsqueeze(dim=1).to(device)
        batch_r = torch.FloatTensor(np.array([transition[2] for transition in sample_batch])).to(device)
        batch_s_ = torch.FloatTensor(np.array([transition[3] for transition in sample_batch])).to(device)

        def critic_learn():
            a1 = self.actor_target(batch_s_).detach()
            y_true = batch_r + GAMMA * self.critic_target(batch_s_, a1).detach()
            y_pred = self.critic(batch_s, batch_a)
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(batch_s, self.actor(batch_s)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

    def save_model(self, actor_path, actor_target_path, critic_path, critic_target_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.actor_target.state_dict(), actor_target_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.critic_target.state_dict(), critic_target_path)

    def load_model(self, actor_path, actor_target_path, critic_path, critic_target_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.actor_target.load_state_dict(torch.load(actor_target_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.critic_target.load_state_dict(torch.load(critic_target_path))
