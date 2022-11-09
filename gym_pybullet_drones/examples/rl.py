from gym_pybullet_drones.examples.cross_tonel import rl
import pybullet as p
import random

env = rl(render=True)

# env.reset()
for _ in range(5):
    
    
    while True:
        
        # p.setRealTimeSimulation(1)  
        print('11111111111111111111111111111')
        run=rl()
        run.run()
        run.ini() 
        act = env.action_space.sample()
        print('随机动作采样',act)
        obs, reward, done, _ = env.step(act=act)
        print("done:",done)
        if done:
            print("break")
            break
            

                             
    # act = env.action_space.sample()         # 在动作空间中随机采样
    # print(act)
    # obs, reward, done, _ = env.step(act)    # 与环境交互
    # print(done)
    # if done == True:
    #     print(done)
    #     env.close()
    #     print('dmibbbbbbbbbbbbbbbb')
    # print('dminaaaaaaaaaaaaaaaaaaaaaaa')
    


# env = crosstonel(render=True)
# obs = env.reset()
# p.setRealTimeSimulation(1)
# while True:
#     act = env.action_space.sample()
#     obs, reward, done, _ = env.step(act)
#     if done:
#         break
# act = env.action_space.sample()
# obs, reward, done, _ = env.step(act)
# print(f"state : {obs}, reward : {reward}")




