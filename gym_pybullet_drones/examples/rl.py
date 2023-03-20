from gym_pybullet_drones.examples.cross_rl import rl_ude
import pybullet as p
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import ddpg

# from stable_baselines3 import DQN
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



env = rl_ude (render=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=60000)
model.save("ppo1_60000")


del model # remove to demonstrate saving and loading
model = PPO.load("ppo1")
# print('ooo')
# vec_env = model.get_env()
x=[]
y=[]
obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(action)
    x.append(action[0])
    #env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()


for j in range(len(x)):
    y.append(j*0.01)
plt.plot(y,x,color='green',linewidth = 3)
plt.show()


