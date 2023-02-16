from gym_pybullet_drones.examples.cross_rl import rl
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



env = rl(render=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=60000)
model.save("ppo1")
del model # remove to demonstrate saving and loading
model = PPO.load("ppo1")
# vec_env = model.get_env()
obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()


# x=[]
# y=[]
# # p.setRealTimeSimulation(1)  

# # env.reset()
# for _ in range(1000):
#         print('11111111111111111111111111111')

        
#         # print('ss',x)
#         act = env.action_space.sample()
#         print('随机动作采样',act)
#         obs, reward, done, _ = env.step(act=act)
#         x.append(obs[0])
#         y.append(obs[1])
#         # print('ss',x)
#         print("done:",done)
#         if done:
#             print("break")
#             break
# # plt.plot(x,y,color='green',linewidth = 3)
# # plt.show()


