import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from gym_pybullet_drones.examples.cross_rl import crosstonel
from stable_baselines3 import A2C
import numpy as np
import pybullet as p


class MySim(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            low=np.array([-10.]),
            high=np.array([10.]),
            dtype=np.float32
        )
        self.spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1]),
                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1]),
                              dtype=np.float32
                              )



    def _applyaction(self,action):
        ude_t = action
        ude_t = np.clip(ude_t, -10., 10.)
        
        agent=crosstonel(ude_t)
        

    def step(self, action):
        state = 1

        if action == 2:
            reward = 1
        else:
            reward = -1
        done = True
        info = {}
        return state, reward, done, info
    
    def reset(self):
        p.resetSimulation(physicsClientId=self._physics_client_id)
        crosstonel()
        return self.__get_observation()
    
    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass
    
    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1
    
if __name__ == "__main__":
    # from stable_baselines import PPO2
    # from stable_baselines import deepq
    env = MySim()
    model = A2C("MlpPolicy",
                    env,
                    verbose=1
                    )
    # model = A2C.DQN(policy="MlpPolicy", env=env)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    # 验证十次
    for _ in range(10):
        action, state = model.predict(observation=obs)
        print(action)
        obs, reward, done, info = env.step(action)
        env.render()