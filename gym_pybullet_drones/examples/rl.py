from gym_pybullet_drones.examples.cross_rl import rl
import pybullet as p
import random
import matplotlib.pyplot as plt

env = rl(render=True)
x=[]
y=[]
# p.setRealTimeSimulation(1)  

# env.reset()
for _ in range(1):
        print('11111111111111111111111111111')

        
        print('ss',x)
        act = env.action_space.sample()
        print('随机动作采样',act)
        obs, reward, done, _ = env.step(act=act)
        x.append(obs[0])
        y.append(obs[1])
        print('ss',x)
        print("done:",done)
        if done:
            print("break")
            break
            

# plt.plot(x,y,color='green',linewidth = 3)
# plt.show()


