from gym_pybullet_drones.examples.cross_rl import rl
import pybullet as p
import random
import matplotlib.pyplot as plt
# from stable_baselines3 import PPO2
from stable_baselines3 import ddpg

env = rl()



episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()           
    done = False
    score = 0

    while not done:
        # env.render()                           # 渲染环境
        action = env.action_space.sample()     # 随机采样动作
        print(action)
        n_state, reward, done, info = env.step(action)    # 和环境交互，得到下一个状态，奖励等信息
        score += reward                        # 计算分数
    print("Episode : {}, Score : {}".format(episode, score))

env.close()     # 关闭窗口




# model = ddpg.ddpg(policy="MlpPolicy", env=env)
# model.learn(total_timesteps=10000)

# obs = env.reset()
# 验证十次
# for _ in range(10000):
#     action, state = model.predict(observation=obs)
#     print(action)
#     obs, reward, done, info = env.step(action)
    # env.render()


# env = rl(render=True)
# x=[]
# y=[]
# # p.setRealTimeSimulation(1)  

# # env.reset()
# for _ in range(1):
#         print('11111111111111111111111111111')

        
#         print('ss',x)
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
            

# plt.plot(x,y,color='green',linewidth = 3)
# plt.show()


