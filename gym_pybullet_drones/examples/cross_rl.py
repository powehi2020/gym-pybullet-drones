"""Script demonstrating the implementation of the downwash effect model.

Example
-------
In a terminal, run as:

    $ python cross_rl.py

Notes
-----
The drones move along 2D trajectories in the X-Z plane, between x == +.5 and -.5.

"""
import time
import argparse
import numpy as np
import pybullet as p
import random

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_AGGREGATE = True
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

class crosstonel(CtrlAviary):
    
    def __init__(self, render : bool = False):
        self._render = render
        # 定义动作空间
        self.action_space = spaces.Box(
            low=np.array([-10.]),
            high=np.array([10.]),
            dtype=np.float32
            )
        
        
        # self.PYB_CLIENT = p.connect(p.GUI if self._render else p.DIRECT)

        # 定义状态空间
        # self.observation_space = spaces.Box(
        #     low=np.array([0., 0.]),
        #     high=np.array([100., np.pi])
        # )
        
        
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  600000, 600000, 600000, 600000])
        self.observation_space  =  spaces.Box(low=obs_lower_bound,
                                            high=obs_upper_bound,
                                            dtype=np.float32
                                            )
        # print(self.observation_space)
        # self.observation_space = spaces.Box(np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
        #                    self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]]))
       
        # 计数器
        self.step_num = 0
        
        
        # self.PYB_CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
        # for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
        #     p.configureDebugVisualizer(i, 0, physicsClientId=self.PYB_CLIENT)
        # p.resetDebugVisualizerCamera(cameraDistance=3,
        #                                 cameraYaw=-30,
        #                                 cameraPitch=-30,
        #                                 cameraTargetPosition=[0, 0, 0],
        #                                 physicsClientId=self.PYB_CLIENT
        #                                 )
        # ret = p.getDebugVisualizerCamera(physicsClientId=self.PYB_CLIENT)
        # print("viewMatrix", ret[2])
        # print("projectionMatrix", ret[3])

    def apply_action(self,
            drone=DEFAULT_DRONE, 
            gui=DEFAULT_GUI, 
            record_video=DEFAULT_RECORD_VIDEO, 
            simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, 
            control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, 
            aggregate=DEFAULT_AGGREGATE, 
            duration_sec=DEFAULT_DURATION_SEC,
            output_folder=DEFAULT_OUTPUT_FOLDER,
            plot=True,
            colab=DEFAULT_COLAB,
            act=0
        ):
        #gui=DEFAULT_GUI
        self.act = act
        self.act = np.clip(self.act, -10., 10.)
        print('apply_action',self.act)
        #### Initialize the simulation #############################
        INIT_XYZS = np.array([[0, 0, random.uniform(.7,1)],[0, 0, random.uniform(.2,.6)]])#飞机的初始位置x，y，z


        AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1
        self.env = CtrlAviary(drone_model=drone,
                        num_drones=2,
                        initial_xyzs=INIT_XYZS,
                        physics=Physics.PYB_DW,
                        neighbourhood_radius=10,
                        freq=simulation_freq_hz,
                        aggregate_phy_steps=AGGR_PHY_STEPS,
                        gui=gui,
                        record=record_video,
                        obstacles=True
                        )
        #### Obtain the PyBullet Client ID from the environment ####
        self.PYB_CLIENT = self.env.getPyBulletClient()

        #### Initialize the trajectories ###########################
        PERIOD = 5
        NUM_WP = control_freq_hz*PERIOD
        TARGET_POS = np.zeros((NUM_WP, 2))
        for i in range(NUM_WP):
            TARGET_POS[i, :] = [np.cos(2*np.pi*(i/NUM_WP)), 0]
        wp_counters = np.array([0, int(NUM_WP/2)])

        #### Initialize the logger #################################
        logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                        num_drones=2,
                        duration_sec=duration_sec,
                        output_folder=output_folder,
                        colab=colab
                        )

        #### Initialize the obstacle ####
  
        p.loadURDF("cube_no_rotation.urdf", [0.7,-1,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#正方体
        p.loadURDF("cube_no_rotation.urdf", [0.7,1,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.PYB_CLIENT)#正方体
        
        

        #### Initialize the controllers ############################
        
        ctrl = [DSLPIDControl(drone_model=drone,ude_t=self.act) for i in range(2)]
        
        
        

        #### Run the simulation ####################################
        CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ/control_freq_hz))
        action = {str(i): np.array([0, 0, 0, 0]) for i in range(2)}
        
        START = time.time()
        for i in range(0, int(duration_sec*self.env.SIM_FREQ), AGGR_PHY_STEPS):
            

            #### Step the simulation ###################################
            self.obs, reward, done, info = self.env.step(action)

            self.obs['1']['state']=np.array([self.obs['1']['state'][0],self.obs['1']['state'][1],self.obs['1']['state'][2],self.obs['1']['state'][3],self.obs['1']['state'][4],self.obs['1']['state'][5],self.obs['1']['state'][6],self.obs['1']['state'][7],self.obs['1']['state'][8],self.obs['1']['state'][9],self.obs['1']['state'][10],self.obs['1']['state'][11],self.obs['1']['state'][12],self.obs['1']['state'][13],self.obs['1']['state'][14],self.obs['1']['state'][15],self.obs['1']['state'][16],self.obs['1']['state'][17],self.obs['1']['state'][18],self.obs['1']['state'][19]])
         

            #### Compute control at the desired frequency ##############
            if i%CTRL_EVERY_N_STEPS == 0:

                #### Compute control for the current way point #############
                for j in range(2):
                    action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*self.env.TIMESTEP,
                                                                        state=self.obs[str(j)]["state"],
                                                                        target_pos=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j, 2]]),
                                                                        )

                #### Go to the next way point and loop #####################
                for j in range(2):
                    wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

            #### Log the simulation ####################################
            for j in range(2):
                logger.log(drone=j,
                        timestamp=i/self.env.SIM_FREQ,
                        state=self.obs[str(j)]["state"],
                        control=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j ,2], np.zeros(9)])
                        )

            #### Printout ##############################################
            # if i%self.env.SIM_FREQ == 0:
            #     self.env.render()
                # print(print(obs[str(1)]["state"]))

            #### Sync the simulation ###################################
            # if gui:
            #     sync(i, START, self.env.TIMESTEP)
                
                
        # print("dnikmn")        
        # self.reset=self.env.reset()

        #### Close the environment #################################
        self.env.close()
        

        #### Save the simulation results ###########################
        # logger.save()
        # logger.save_as_csv("dw") # Optional CSV save

        ### Plot the simulation results ###########################
        # if plot :
        #     logger.plot()
            
    def __get_observation(self):
        
        state=self.obs['1']['state']
        print('uuuuuuuuuuuu',state)
       
        return np.array(state, dtype=np.float32)
    
    
    def reset(self):
        self.apply_action(
            drone=DEFAULT_DRONE, 
            gui=False, 
            record_video=DEFAULT_RECORD_VIDEO, 
            simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, 
            control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, 
            aggregate=DEFAULT_AGGREGATE, 
            duration_sec=DEFAULT_DURATION_SEC,
            output_folder=DEFAULT_OUTPUT_FOLDER,
            plot=True,
            colab=DEFAULT_COLAB,
            act=0
        )
        # p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        # self.env._housekeeping()
        # self.env._updateAndStoreKinematicInformation()
        # self.env._startVideoRecording()
        
        
        return self.__get_observation()
    
    
    def step(self, act):
        self.act=act
        print('aaaaaaaaaaaaaaaaaaaaaa',self.act)
        self.apply_action(act=self.act)

        
        
        self.step_num += 1
        state = self.__get_observation()
        reward = random.random()
        print('lllllllllllllllllll',self.step_num)
        if  self.step_num > 5:
            done = True
        else:
            done = False
        info = {}
        return state, reward, done, self.act
        
    def close (self):
        self.env.close()
        
    def render(self, mode='human'):
        pass

    
