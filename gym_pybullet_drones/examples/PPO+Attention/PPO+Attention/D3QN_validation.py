#!/home/dengyu/.conda/envs/py37/bin/python3.7

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import APFSimulate
import random
import os

from ruamel.yaml import YAML, dump, RoundTripDumper
import os,time,subprocess
from flightgym import QuadrotorEnv_v1

# change work directory to find the model file
os.chdir(os.path.dirname(__file__))
# print(os.getcwd())
'''
cfg = YAML().load(open("flightenv.yaml", 'r'))
cfg["env"]["render"] = "yes"
unity = subprocess.Popen(os.environ["FLIGHTMARE_PATH"] +"/flightrender/RPG_Flightmare.x86_64")
fenv = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
fenv.connectUnity()
'''

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
gamma = 0.99  # discount factor
# random seed
# randomseed = 69963
randomseed = 1110
torch.manual_seed(randomseed)
np.random.seed(randomseed)
random.seed(randomseed)

env = APFSimulate.APFSimulator(gamma, 'Valid')  # instantiate the environment
num_state = env.num_state  # the dimension of state space
num_action = env.num_action  # the number of discretized actions


class Net(nn.Module):
    '''
    The network class, pytorch==1.8.
    Class methods:
        __init__: initialization function, 6 fully-connected layers
        forward: forward propagation
    Ref:Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning.
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128 + 5, 128)
        # self.layer2 = nn.Linear(5, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, num_action)
        self.layer6 = nn.Linear(64, 1)

    def forward(self, x):
        '''
        Input:
            x: observations
        output:
            action_value: Q value for each action
        '''
        state_loc = x[:, :5]
        # embedding
        state_ij = x[:, 5:]
        x = state_ij.reshape(-1, 2)
        x = torch.relu(self.layer1(x))
        feature = torch.vstack((x[0::2, :].reshape(1, -1), x[1::2, :].reshape(1, -1)))
        
        # concatenate
        if state_ij.shape[1] != 0:
            mean_feature = torch.nanquantile(feature, 0.5, dim=0).reshape(-1, x.shape[1])  # calculate mean of features
            mean_feature = torch.hstack((state_loc, mean_feature))
        else:
            mean_feature = state_loc
        # calculate advantage
        x = torch.relu(self.layer2(mean_feature))
        advantage = torch.relu(self.layer3(x))
        advantage = self.layer5(advantage)
        # calculate state value
        state_value = torch.relu(self.layer4(x))
        state_value = self.layer6(state_value)
        # calculate Q value
        action_value = state_value + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return action_value

fstate = np.zeros((1,13), np.float32)
fstate[0,3] = 1
fstate[0,0] = 0
fstate[0,2] = 1
frame = 0

net = Net()  # instantiate the network
net.load_state_dict(torch.load('6000.pt'))  # load parameters
state = env.reset()  # reset the environment
last_done = np.array([[0., 0, 0]])  # if pursuers are inactive at the last timestep
while True:
    env.render(False)  # render the pursuit process
    fstate[0, 0] = env.v1.position[0] / 500
    fstate[0, 1] = env.v1.position[1] / 500
    # fenv.setState(fstate, frame)
    frame += 1
    # plt.savefig(str(j)) # save figures
    action = np.zeros((1, 0))  # action buffer
    # choose actions
    for i in range(env.num_agent):
        temp = state[:, i:i + 1].reshape(1, -1)
        temp = net(torch.tensor(np.ravel(temp), dtype=torch.float32).view(1, -1))
        action_temp = torch.max(temp, 1)[1].data.numpy()
        action = np.hstack((action, np.array(action_temp, ndmin=2)))
    # execute action
    state_next, reward, done = env.step(action)
    temp1 = done == 1
    temp2 = done == 2
    temp = np.vstack((temp1, temp2))
    if np.all(np.any(temp, axis=0, keepdims=True)):
        # if all pursuers capture the evader or the episode reaches maximal length
        break
    state = state_next
    last_done = done
    