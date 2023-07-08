"""Script demonstrating the implementation of the downwash effect model.

Example
-------
In a terminal, run as:

    $ python downwash.py

Notes
-----
The drones move along 2D trajectories in the X-Z plane, between x == +.5 and -.5.

"""
import time
import argparse
import numpy as np
import pybullet as p
import random
from gym import spaces
import gym
import time

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.DSLPIDControl_old import DSLPIDControl_old
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_AGGREGATE = True
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


class rl_ude (CtrlAviary,gym.Env):
             
        
        
    def __init__(self,
            drone=DEFAULT_DRONE, 
            gui=True, 
            record_video=DEFAULT_RECORD_VIDEO, 
            simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, 
            control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, 
            aggregate=DEFAULT_AGGREGATE, 
            duration_sec=DEFAULT_DURATION_SEC,
            output_folder=DEFAULT_OUTPUT_FOLDER,
            plot=True,
            colab=DEFAULT_COLAB,
            render : bool = False
        ):
        self._render = render
        # 定义动作空间
        self.action_space = spaces.Box(
            low=np.array([0.01]),
            high=np.array([50.]),
            dtype=np.float32
            )
        # self.self.PYB_CLIENT = p.connect(p.GUI if self._render else p.DIRECT)

        # 定义状态空间
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  600000, 600000, 600000, 600000])
        self.observation_space  =  spaces.Box(low=obs_lower_bound,
                                            high=obs_upper_bound,
                                            dtype=np.float32
                                            )
       
        #### Initialize the simulation #############################
        self.INIT_XYZS = np.array([[1, 0, 0.6],[-1, 0 , 1]])#飞机的初始位置x，y，z  random.uniform(0.4,0.8)
        self.AGGR_PHY_STEPS  = int(simulation_freq_hz/control_freq_hz) if aggregate else 1
        self.env = CtrlAviary(drone_model=drone,
                        num_drones=2,
                        initial_xyzs=self.INIT_XYZS,
                        physics=Physics.PYB_DW,
                        neighbourhood_radius=10,
                        freq=simulation_freq_hz,
                        aggregate_phy_steps=self.AGGR_PHY_STEPS ,
                        gui=gui,
                        record=record_video,
                        obstacles=True
                        )
        
        self.step_num = 0
        self.PYB_CLIENT = self.env.getPyBulletClient()
        
        
        

        #### Initialize the trajectories ###########################
        PERIOD = 20
        self.NUM_WP = control_freq_hz*PERIOD
        self.TARGET_POS = np.zeros((self.NUM_WP, 2))
        for i in range(self.NUM_WP):
            self.TARGET_POS[i, :] = [np.cos(2*np.pi*(i/self.NUM_WP)), 0]
           

        self.wp_counters = np.array([0, int(self.NUM_WP/2)])

        #### Initialize the logger #################################
        self.logger = Logger(logging_freq_hz=int(simulation_freq_hz/self.AGGR_PHY_STEPS ),
                        num_drones=2,
                        duration_sec=duration_sec,
                        output_folder=output_folder,
                        colab=colab
                        )

        #### Initialize the obstacle ####
        """
        p.loadURDF("urdf文件", xyz坐标, 欧拉角), 仿真环境的ID)
        urdf文件:文件路径:/home/lkder/anaconda3/envs/drones/lib/python3.8/site-packages/pybullet_data
        xyz坐标:[0,0,0]
        欧拉角:p.getQuaternionFromEuler([0,0,0]
        仿真环境的ID:physicsClientId=self.PYB_CLIENT
        例子:p.loadURDF("sphere2.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)# 球
            将sphere2,以[0,0,0]位置,p.getQuaternionFromEuler([0,0,0]角度,置于环境self.PYB_CLIENT中
        """
        # p.loadURDF("sphere2.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)# 球
        # p.loadURDF("duck_vhacd.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#鸭子
        # p.loadURDF("cube_no_rotation.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#正方体
        # p.loadURDF("samurai.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#啥也没有！！
        # p.loadURDF("soccerball.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#足球
        # p.loadURDF("teddy_vhacd.urdf", [0,0,0], p.getQuaternionFromEuler([90,0,0]), physicsClientId=self.PYB_CLIENT)#小熊

        p.loadURDF("cube_no_rotation.urdf", [0.7,-1,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#正方体
        p.loadURDF("cube_no_rotation.urdf", [0.7,1,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#正方体

        
        #### Initialize the controllers ############################
       

        self.ctrl = [DSLPIDControl(drone_model=DroneModel('cf2x')) for i in range(1)]
        self.ctrl1 = [DSLPIDControl_old(drone_model=DroneModel('cf2x')) for i in range(1)]

        #### Run the simulation ####################################
        self.CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ/48))
        self.action = {str(i): np.array([0, 0, 0, 0]) for i in range(2)}
        
        self.START = time.time()
         
        
    def step (self,act=None):
        #### Step the simulation ###################################
        self.obs, reward, dones, info = self.env.step(self.action)
        self.state = np.array([self.obs['1']['state'][0],self.obs['1']['state'][1],self.obs['1']['state'][2],self.obs['1']['state'][3],self.obs['1']['state'][4],self.obs['1']['state'][5],self.obs['1']['state'][6],self.obs['1']['state'][7],self.obs['1']['state'][8],self.obs['1']['state'][9],self.obs['1']['state'][10],self.obs['1']['state'][11],self.obs['1']['state'][12],self.obs['1']['state'][13],self.obs['1']['state'][14],self.obs['1']['state'][15],self.obs['1']['state'][16],self.obs['1']['state'][17],self.obs['1']['state'][18],self.obs['1']['state'][19]], dtype=np.float32)
        ### 下方无人机的控制器    
        self.action[str(0)], _, _ = self.ctrl[0].computeControlFromState(control_timestep=self.CTRL_EVERY_N_STEPS*self.env.TIMESTEP,
                                                                state=self.obs[str(0)]["state"],
                                                                target_pos=np.hstack([self.TARGET_POS[self.wp_counters[0], :], self.INIT_XYZS[0, 2]]),
                                                                ude=act)    
        ### 上方无人机的控制器 
        self.action[str(1)], _, _ = self.ctrl1[0].computeControlFromState(control_timestep=self.CTRL_EVERY_N_STEPS*self.env.TIMESTEP,
                                                                state=self.obs[str(1)]["state"],
                                                                target_pos=np.hstack([self.TARGET_POS[self.wp_counters[1], :], self.INIT_XYZS[1, 2]]),
                                                                ude=None) 
        rewards = self.ctrl[0].compute_reward()
        done = self.ctrl[0].compute_done()

        #### Go to the next way point and loop #####################
        for j in range(2):
            self.wp_counters[j] = self.wp_counters[j] + 1 if self.wp_counters[j] < (self.NUM_WP-1) else 0

            # #### Sync the simulation ###################################
            if True:
                sync(j, self.START, self.env.TIMESTEP)
                # print('时间',self.START)


            
        return self.state, rewards, done, info
    
       
    def close (self):
        self.env.close()
        
        
    def reset (self):
        # self.obs, reward, dones, info = self.env.step(self.action)

        self.obs = self.env.reset()
        self.wp_counters = np.array([0,int(self.NUM_WP/2)])
        p.loadURDF("cube_no_rotation.urdf", [0.7,-1,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#正方体
        p.loadURDF("cube_no_rotation.urdf", [0.7,1,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#正方体

        # print(self.obs,'sss')
        return self.obs[str(1)]["state"]
        
    
        


