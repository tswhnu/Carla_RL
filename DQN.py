import random
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
import os
from collections import deque
import math

# the defination of the hyper parameters
BATCH_SIZE = 16  # batch size of the training data
LR = 0.001  # learning rate
EPSILON = 0.6  # greedy algorithm
GAMMA = 0.9  # reward discount
TARGET_UPDATE = 100  # update the target network after training
MEMORY_CAPACITY = 100  # the capacity of the memory
N_STATE = 4  # the number of states that can be observed from environment
N_ACTION = 3  # the number of action that the agent should have
N_CHANNEL = 6
# decide the device used to train the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the structure of the network

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000


class Linear_Net(nn.Module):
    def __init__(self, n_state=N_STATE, n_action=N_ACTION):
        super(Linear_Net, self).__init__()
        self.fc1 = nn.Linear(n_state, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, n_action)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


class CNN(nn.Module):
    def __init__(self, n_channel=N_CHANNEL, out_channel=N_ACTION):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channel, out_channels=16, kernel_size=3, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(50 * 50 * 64, 256)
        self.fc2 = nn.Linear(256, out_channel)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float().to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        out = self.tanh(self.fc2(x))

        return out


class DQN(object):
    def __init__(self, test=False, var_eps=True):
        self.policy_net, self.target_net = Linear_Net().to(device), Linear_Net().to(device)
        self.learn_step = 0
        self.memory_counter = 0
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.eval_model_load_path = './model_02/policy_net1000.pt'
        self.target_model_load_path = './model_02/policy_net1000.pt'
        self.eval_model_save_path = './model_evalV2.pt'
        self.target_model_save_path = './model_targetV2.pt'
        self.test_mode = test
        self.var_eps = var_eps

    def select_action(self, state):

        state = torch.unsqueeze(torch.tensor(state), dim=0)
        p = np.random.random()

        if os.path.exists(self.eval_model_load_path):
            E_thresh = EPS_END
        else:
            E_thresh = EPS_END + (EPS_START - EPS_END) * \
                       math.exp(-1. * self.learn_step / EPS_DECAY)
        if self.test_mode:
            actions_value = self.policy_net.forward(state)
            return torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:
            if p > E_thresh:
                actions_value = self.policy_net.forward(state)
                return torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            else:
                return np.random.randint(0, N_ACTION)

    def store_transition(self, s, a, r, s_, done):
        transition = [s, a, r, s_, done]
        self.memory.append(transition)
        self.memory_counter += 1

    def optimize_model(self):

        if self.learn_step % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_step += 1

        # get the samples to train the policy net
        sample_batch = random.sample(self.memory, BATCH_SIZE)
        batch_s = torch.FloatTensor(np.array([transition[0] for transition in sample_batch])).to(device)
        batch_a = torch.LongTensor(np.array([transition[1] for transition in sample_batch])).unsqueeze(dim=1).to(device)
        batch_r = torch.FloatTensor(np.array([transition[2] for transition in sample_batch])).to(device)
        batch_s_ = torch.FloatTensor(np.array([transition[3] for transition in sample_batch])).to(device)

        # calculate the q_value
        q_eval = self.policy_net(batch_s).gather(1, batch_a)
        q_next = self.target_net(batch_s_).max(1)[
            0].detach()  # use detach to avoid the backpropagation during the training
        q_target = []
        for index, (s, a, r, s_, done) in enumerate(sample_batch):
            if not done:
                q_target_value = batch_r.squeeze()[index] + GAMMA * q_next[index]
            else:
                q_target_value = batch_r.squeeze()[index]
            q_target.append(q_target_value)
        q_target = torch.tensor(q_target).to(device)
        loss = self.loss_func(q_eval.squeeze(), q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, policy_path, target_path):
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.target_net.state_dict(), target_path)

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.eval_model_load_path))
        self.target_net.load_state_dict(torch.load(self.target_model_load_path))
