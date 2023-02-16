import math, os
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# from IPython.display import clear_output
import matplotlib.pyplot as plt
# %matplotlib inline

# change work directory to find the model file
# os.chdir(os.path.dirname(__file__))
# print(os.getcwd())

# Use CUDA
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# random seed
torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)

# Create Environments
'''
from common.multiprocessing_env import SubprocVecEnv
num_envs = 16
env_name = "Pendulum-v0"

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)
'''
import APFSimulate
gamma = 0.99  # the discount factor
env = APFSimulate.APFSimulator(gamma, 'Train')  # instantiate the environment
envtest = APFSimulate.APFSimulator(gamma, 'Valid')
# envtest = env

class RunningStat:
    '''
    Calculate the mean and std of all previous rewards.
    Class methods:
        __init__: the initialization function
        push: update statistics
    '''

    def __init__(self):
        '''
        The initialization function
        '''
        self.n = 0  # the number of reward signals collected
        self.mean = np.zeros((1,))  # the mean of all rewards
        self.s = np.zeros((1,))
        self.std = np.zeros((1,))  # the std of all rewards

    def push(self, x):
        '''
        Update statistics.
        Input:
            x: the reward signal
        '''
        self.n += 1  # update the number of reward signals collected
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n  # update mean
            self.s = self.s + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.s / (self.n - 1) if self.n > 1 else np.square(self.mean))  # update stm

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


def plot(frame_idx, rewards):
    # clear_output(True)
    # plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.figure(2)
    plt.clf()
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show(block=False)
    plt.pause(0.001)
    # plt.figure(1)
    
def test_env(vis=False):
    state = envtest.reset()
    if vis: envtest.render()
    done = False
    total_reward = 0
    tempframe = 0
    while not np.all(done) and tempframe < 1000:
        # env.render(False)  # render the training process
        action = np.zeros((num_outputs, envtest.num_agent))  # action index buffer

        for i in range(envtest.num_agent):
            istate = state[:, i:i + 1].reshape(1, -1)
            istate = torch.FloatTensor(istate).to(device)
            dist, _ = model(istate)
            action_temp = dist.sample().cpu().numpy()[0]
            action[:, i] = action_temp.reshape(1,-1)
        next_state, reward, done = envtest.step_ppo(action)
        state = next_state
        tempframe += 1
        if tempframe % 4 == 0 and vis:
            envtest.render(reward=total_reward, action=action)
        total_reward += reward
    return total_reward


# GAE tau=0:TD, tau=1:Monte Carlo
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.9):
    values = torch.hstack((values, next_value))
    gae = 0
    returns = torch.zeros((rewards.shape[0], 1)).to(device)
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns[step, 0] = gae + values[step]
        # returns.insert(0, gae + values[step])
    return returns

# Proximal Policy Optimization Algorithm
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    # for x in range(batch_size // mini_batch_size -1, -1, -mini_batch_size):
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        # yield states[x:x+1, :], actions[x:x+1, :], log_probs[x:x+1, :], returns[x:x+1, :], advantage[x:x+1, :]

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# num_inputs  = envs.observation_space.shape[0]
# num_outputs = envs.action_space.shape[0]
num_inputs = 128
num_outputs = 2

#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 100
mini_batch_size  = 20
ppo_epochs       = 4
threshold_reward = 100000

running_stat = RunningStat()
model = ActorCritic(num_inputs, num_outputs, hidden_size, std=0).to(device)
# model.load_state_dict(torch.load('ppo100000.pt'))
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 10000000
frame_idx  = 0
goals = []
train_returns = torch.zeros((0,1))

state = env.reset()
early_stop = False

refresh_step = 0
lastdone = np.zeros((1, env.num_agent))
goal = 0

# while frame_idx < max_frames and not early_stop:
while frame_idx < max_frames:

    log_probs = torch.zeros((0, env.num_agent * num_outputs)).to(device)
    values    = torch.zeros((0, env.num_agent)).to(device)
    states    = torch.zeros((0, env.num_agent * env.num_state)).to(device)
    actions   = torch.zeros((0, env.num_agent * num_outputs)).to(device)
    rewards   = torch.zeros((0, env.num_agent)).to(device)
    masks     = torch.zeros((0, env.num_agent)).to(device)
    entropy = 0

    for step in range(num_steps):
        action3 = torch.zeros((1, 0)).to(device)  # action index buffer
        value3 = torch.zeros((1, 0)).to(device)
        log3 = torch.zeros((1, 0)).to(device)
        state3 = torch.zeros((1, 0)).to(device)
        for i, v in enumerate(env.vehicles):
            istate = state[:, i:i + 1].reshape(1, -1)
            istate = torch.FloatTensor(istate).to(device)
            dist, value_temp = model(istate)
            action_temp = dist.sample()

            log_prob = dist.log_prob(action_temp)
            entropy += dist.entropy().mean()
            action3 = torch.hstack((action3, action_temp))
            log3 = torch.hstack((log3, log_prob))
            value3 = torch.hstack((value3, value_temp))
            state3 = torch.hstack((state3, istate))

        action = action3.cpu()
        action = torch.vstack((action[:, 0::2], action[:, 1::2]))
        next_state, reward, done = env.step_ppo(action.numpy())
        
        # if step % 20 == 0:
            # env.render(reward=reward, action=action.cpu().numpy())

        mask3 = torch.FloatTensor(done).to(device)
        for i in range(mask3.shape[1]):
            if mask3[0, i] > 1:
                mask3[0, i] = 0
            mask3[0, i] = mask3[0, i] - 1
            running_stat.push(reward[:, i])
            # reward[0, i] = np.clip(reward[0, i] / (running_stat.std + 1e-8), -10, 10)  # reward normalization

        states = torch.vstack((states, state3))
        actions = torch.vstack((actions, action3))
        rewards = torch.vstack((rewards, torch.FloatTensor(reward).to(device)))
        masks = torch.vstack((masks, mask3))
        values = torch.vstack((values, value3))
        log_probs = torch.vstack((log_probs, log3))
        
        state = next_state
        frame_idx += 1
        
        goal = goal * 0.99 + np.sum(reward * np.logical_not(lastdone))

        if np.any(np.logical_xor(done, lastdone)):
            env.render(reward=reward, action=action.numpy())
            break

        if env.t >= 1000:
            env.render(reward=reward, action=action.numpy())
            env.reset()
            finish_flag = True
            print("Failed at frame " + str(frame_idx) + " step " + str(refresh_step))
            break
        
        if frame_idx % 50000 == 0:
            string = 'ppo'+str(frame_idx) + '.pt'
            torch.save(model.state_dict(), string)
            # env.render(reward=reward, action=action.numpy())

    refresh_step += 1
    next_state = torch.FloatTensor(next_state).to(device)
    losssort = [x for x in range(env.num_agent)]
    # losssort = np.delete(losssort, lastdone.astype(bool).ravel().tolist())
    random.shuffle(losssort)
    for i in losssort:
    # if True:
        if not lastdone[0, i]:
            # i = random.randint(0, env.num_agent-1)
            istate = next_state[:, i:i + 1].reshape(1, -1).to(device)
            # istate = torch.FloatTensor(istate).to(device)
            _, next_value = model(istate)
            preturns = compute_gae(next_value[0], rewards[:,i], masks[:, i], values[:, i]).detach()
            # preturns   = torch.cat(preturns).detach().reshape(-1, 1)
            plog_probs = log_probs[:, num_outputs*i:num_outputs*(i+1)].detach()
            pvalues    = values[:, i:i+1].detach()
            pstates    = states[:, env.num_state*i:env.num_state*(i+1)].detach()
            pactions   = actions[:, num_outputs*i:num_outputs*(i+1)].detach()
            padvantages = preturns - pvalues

            # PPO Update
            ppo_update(ppo_epochs, mini_batch_size, pstates, pactions, plog_probs, preturns, padvantages)

            # train_returns = torch.vstack((train_returns, preturns.cpu()))
    
    lastdone = done
    if np.all(done == 1):
        goals.append(goal)
        env.reset()
        lastdone = np.zeros((1, env.num_agent))
        print("Successed at frame " + str(frame_idx) + " step " + str(refresh_step))
        plot(frame_idx, goals)
        np.savetxt("goals.txt", goals)

plt.show()