import gym
from gym_pybullet_drones.examples.cross_rl import rl_ude
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

Step = 50000

env = rl_ude (render=False)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo/")
# model.learn(total_timesteps=10000000)
# model.save("ppo_cartpole4")
del model # remove to demonstrate saving and loading

model = PPO.load("/home/lkd/Documents/GitHub/gym-pybullet-drones/ppo_cartpole4.zip")
obs = env.reset()
x=[]
y=[]
re = []
trajectory = []
# np.append(trajectory_des,values = obs , axis=0)
obs = env.reset()
for i in range(Step):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    trajectory.append([obs[7],obs[8],obs[9]])
    # print(obs[0])
    x.append(action[1])
    re.append(reward)
    if done:
      obs = env.reset()

env.close()

# print(len(y),len(trajectory[:0]),'###')
trajectory = np.array(trajectory)
# print(trajectory)
for j in range(len(re)):
    y.append(j*0.01)

plt.subplot(2, 2, 2)
plt.plot(y,x,color='green',linewidth = 1)
plt.title("T_ude")
plt.xlim(0,10)


plt.subplot(2, 2, 4)
plt.plot(y,re,color='r',linewidth = 1)
plt.title("reward")
plt.xlim(0,10)


# print(trajectory[:,0],trajectory,'jjj')
plt.subplot(3, 2, 1)
plt.plot(y,trajectory[:,0],color='y',linewidth = 1)
plt.title("r")
plt.xlim(0,10)
# plt.show()

plt.subplot(3, 2, 3)
plt.plot(y,trajectory[:,1],color='b',linewidth = 1)
plt.title("p")
plt.xlim(0,10)
# plt.show()

plt.subplot(3, 2, 5)
plt.plot(y,trajectory[:,2],color='black',linewidth = 1)
plt.title("q")
plt.xlim(0,10)
# plt.ylim(0.,0.9)
plt.show()
