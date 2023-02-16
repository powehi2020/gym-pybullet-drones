import APFSimulate, os, torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal

# from IPython.display import clear_output
import matplotlib.pyplot as plt
# %matplotlib inline

# change work directory to find the model file
os.chdir(os.path.dirname(__file__))
# print(os.getcwd())

# Use CUDA
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Create Environments
import APFSimulate
gamma = 0.99  # the discount factor
env = APFSimulate.APFSimulator(gamma, 'Valid')  # instantiate the environment

# Neural Network
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            # nn.ReLU(),
            # nn.Linear(64, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            # nn.ReLU(),
            # nn.Linear(64, num_outputs)
            # nn.Sigmoid()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128 + 5, 128)

        self.embbedLayer = nn.Linear(2, 128)
        self.hlayer = nn.Linear(128, 128)
        self.attentionLayer = nn.Linear(128 * 2, 1)

        self.apply(init_weights)
    
    def forward(self, x):
        return self.forward_attention(x)

    def forward_attention(self, x):
        nextx = torch.zeros((0, num_inputs)).to(device)
        for i in range(x.shape[0]):
            state_loc = x[i:i+1, :5]
            state_ij = x[i:i+1, 5:]
            # embedding
            temp = state_ij.reshape(-1, 2)
            e = torch.relu(self.embbedLayer(temp))
            em = torch.mean(e, dim=0).repeat((e.shape[0], 1))
            h = torch.relu(self.hlayer(e))
            attention_score = torch.relu(self.attentionLayer(torch.hstack((e,em))))
            feature = torch.softmax(attention_score, dim=0) * h
            mean_feature = torch.mean(feature, dim=0, keepdim=True)
            temp = torch.hstack((state_loc, mean_feature))

            temp = torch.relu(self.layer2(temp))
            nextx = torch.vstack((nextx, temp))
        # X = L2(L1(state_loc) + embedded_ij)

        value = self.critic(nextx)
        mu    = self.actor(nextx)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        # dist = mu
        return dist, value
        
    def forward_mean(self, x):
        tempx = torch.zeros((0, num_inputs)).to(device)
        for i in range(x.shape[0]):
            state_loc = x[i:i+1, :5]
            state_ij = x[i:i+1, 5:]
            # embedding
            temp = state_ij.reshape(-1, 2)
            temp = torch.relu(self.layer1(temp))
            # feature = torch.vstack((temp[0::2, :].reshape(1, -1), temp[1::2, :].reshape(1, -1)))
            feature = temp
            mean_feature = torch.mean(feature, dim=0, keepdim=True)
            # concatenate
            mean_feature = torch.hstack((state_loc, mean_feature))

            temp = torch.relu(self.layer2(mean_feature))
            tempx = torch.vstack((tempx, temp))
            # X = L2(L1(state_loc) + embedded_ij)

        value = self.critic(tempx)
        mu    = self.actor(tempx)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        # dist = mu
        return dist, value


def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not np.all(done):
        # env.render(False)  # render the training process
        action = np.zeros((num_outputs, env.num_agent))  # action index buffer

        for i in range(env.num_agent):
            istate = state[:, i:i + 1].reshape(1, -1)
            istate = torch.FloatTensor(istate).to(device)
            dist, _ = model(istate)
            action_temp = dist.sample().cpu().numpy()[0]
            action[:, i] = action_temp.reshape(1,-1)
        next_state, reward, done = env.step_ppo(action)
        state = next_state
        if vis: env.render(reward=total_reward, action=action)
        total_reward += reward
    return total_reward

# num_inputs  = envs.observation_space.shape[0]
# num_outputs = envs.action_space.shape[0]
num_inputs = 128
num_outputs = 2

#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 20
mini_batch_size  = 5
ppo_epochs       = 4
threshold_reward = 1000

model = ActorCritic(num_inputs, num_outputs, hidden_size, std=-5).to(device)
model.load_state_dict(torch.load('ppo500000.pt'))

max_frames = 15000
frame_idx  = 0
test_rewards = []

# state = env.reset()
early_stop = False

test_env(True)
plt.show()

'''
# Saving trajectories for GAIL
from itertools import count

max_expert_num = 50000
num_steps = 0
expert_traj = []

for i_episode in count():
    state = env.reset()
    done = False
    total_reward = 0
    
    while not np.all(done):
        # env.render(False)  # render the training process
        action = np.zeros((1, env.num_agent))  # action index buffer

        for i in range(env.num_agent):
            istate = state[:, i:i + 1].reshape(1, -1)
            istate = torch.FloatTensor(istate).to(device)
            dist, _ = model(istate)
            action_temp = dist.sample().cpu().numpy()[0]
            action[0, i] = action_temp
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        expert_traj.append(np.hstack([state, action]))
        num_steps += 1
    
    print("episode:", i_episode, "reward:", total_reward)
    
    if num_steps >= max_expert_num:
        break
        
expert_traj = np.stack(expert_traj)
print()
print(expert_traj.shape)
print()
np.save("expert_traj.npy", expert_traj)

'''