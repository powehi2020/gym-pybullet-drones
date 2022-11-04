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

class crosstonel():
    def run(
            
            drone=DEFAULT_DRONE, 
            gui=DEFAULT_GUI, 
            record_video=DEFAULT_RECORD_VIDEO, 
            simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, 
            control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, 
            aggregate=DEFAULT_AGGREGATE, 
            duration_sec=DEFAULT_DURATION_SEC,
            output_folder=DEFAULT_OUTPUT_FOLDER,
            plot=True,
            colab=DEFAULT_COLAB
        ):
        #### Initialize the simulation #############################
        INIT_XYZS = np.array([[0, 0, random.uniform(.7,1)],[0, 0, random.uniform(.2,.6)]])#飞机的初始位置x，y，z


        AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1
        env = CtrlAviary(drone_model=drone,
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
        PYB_CLIENT = env.getPyBulletClient()
        
        
        

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
  
        p.loadURDF("cube_no_rotation.urdf", [0.7,-1,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=PYB_CLIENT)#正方体
        p.loadURDF("cube_no_rotation.urdf", [0.7,1,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=PYB_CLIENT)#正方体

        #### Initialize the controllers ############################
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(2)]
        
        
        

        #### Run the simulation ####################################
        CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
        action = {str(i): np.array([0, 0, 0, 0]) for i in range(2)}
        
        START = time.time()
        for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
            

            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)


            #### Compute control at the desired frequency ##############
            if i%CTRL_EVERY_N_STEPS == 0:

                #### Compute control for the current way point #############
                for j in range(2):
                    action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                        state=obs[str(j)]["state"],
                                                                        target_pos=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j, 2]]),
                                                                        )

                #### Go to the next way point and loop #####################
                for j in range(2):
                    wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

            #### Log the simulation ####################################
            for j in range(2):
                logger.log(drone=j,
                        timestamp=i/env.SIM_FREQ,
                        state=obs[str(j)]["state"],
                        control=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j ,2], np.zeros(9)])
                        )

            #### Printout ##############################################
            if i%env.SIM_FREQ == 0:
                env.render()
                # print(print(obs[str(1)]["state"]))

            #### Sync the simulation ###################################
            if gui:
                sync(i, START, env.TIMESTEP)

        #### Close the environment #################################
        # env.close()

        #### Save the simulation results ###########################
        logger.save()
        logger.save_as_csv("dw") # Optional CSV save

        #### Plot the simulation results ###########################
        if plot:
            logger.plot()


if __name__ == "__main__":
    crosstonel.run()