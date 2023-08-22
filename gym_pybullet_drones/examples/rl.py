from gym_pybullet_drones.examples.cross_rl import rl_ude
import pybullet as p
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
env = rl_ude (render=True)
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=25000)
model.save("ppo_cartpole")
Step = 60000
del model # remove to demonstrate saving and loading
model = PPO.load("ppo_cartpole")
vec_env = model.get_env()

x=[]
y=[]
re = []
obs = env.reset()
for i in range(Step):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(action[0],"ude")
    x.append(action[0])
    re.append(reward)
    if done:
      obs = env.reset()

env.close()


for j in range(len(re)):
    y.append(j*0.01)

plt.subplot(1, 2, 1)
plt.plot(y,x,color='green',linewidth = 3)
plt.title("T_ude")


plt.subplot(1, 2, 2)
plt.plot(y,re,color='green',linewidth = 3)
plt.title("reward")
plt.show()





