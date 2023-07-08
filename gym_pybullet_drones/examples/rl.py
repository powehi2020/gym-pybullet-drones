from gym_pybullet_drones.examples.cross_rl import rl_ude
import pybullet as p
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


# from stable_baselines3 import DQN
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



env = rl_ude (render=True)
model = PPO("MlpPolicy", env, verbose=1)

Step = 600000
# model.learn(total_timesteps=Step)
# model.save("ppo_600000_20230407")


del model # remove to demonstrate saving and loading
model = PPO.load("ppo_600000")
# print('ooo')
vec_env = model.get_env()

x=[]
y=[]
re = []
obs = env.reset()
for i in range(Step):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(action)
    x.append(action[0])
    re.append(reward)
    # print(reward,'rew')
    # env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

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





