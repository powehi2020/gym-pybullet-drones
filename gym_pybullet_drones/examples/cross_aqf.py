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


from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.control.DSLPIDControl_old import DSLPIDControl_old
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from scipy.spatial.transform import Rotation as R
# from proplot import rc

from stable_baselines3 import PPO
 
def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler
 



DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_AGGREGATE = True
DEFAULT_DURATION_SEC = 50
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

trajectory_des = []
trajectory_real = []

def Afq (posion):
    Afq = [0,0]
    # obstle = np.array([[-2.1,np.exp(-2.1)+0.7],[2.1,-np.exp(-2.1)-0.7]])
    # for i in np.arange(-2,4,0.1):
    #     obstle=np.append(obstle, [[i,np.exp(-i)+1]],axis = 0)
    #     obstle=np.append(obstle, [[i,-np.exp(-i)-1]],axis = 0)
    
    # for i in np.arange(-4,2,0.1):
    #     obstle=np.append(obstle, [[i+8,np.exp(i)+1]],axis = 0)
    #     obstle=np.append(obstle, [[i+8,-np.exp(i)-1]],axis = 0)
    
    obstle = np.array([[-0,1.5],[-0,-1.5]])
    for i in np.arange(0,1,0.1):
        obstle=np.append(obstle, [[0,-2.5+i],[0,2.5-i]],axis = 0)

    for i in np.arange(0.1,2,0.1):
        obstle=np.append(obstle, [[i,1.5],[i,-1.5]],axis = 0)

    for i in np.arange(2,2.5,0.1):
        obstle=np.append(obstle, [[2,-i+0.5],[2,i-0.5]],axis = 0)

    for i in np.arange(2.1,6,0.1):
        obstle=np.append(obstle, [[i,2],[i,-2]],axis = 0)

    for i in np.arange(4.1,4.8,0.1):
        obstle=np.append(obstle, [[6,i-2.8],[6,-i+2.8]],axis = 0)

    for i in np.arange(6,10,0.1):
        obstle=np.append(obstle, [[i,1.2],[i,-1.2]],axis = 0)


    sigma = 1
    Ck = 3.5
    for i in range(len(obstle)):
        Afq_vetor =  posion - obstle[i] 
        Afq_size = np.linalg.norm(Afq_vetor,ord=2) #二范数
        V_2 = Ck*1/Afq_size*np.exp(-Afq_size**2/2*np.square(sigma))*Afq_vetor
        Afq = Afq + V_2

      
    
    return -1*Afq
    # return [0,0]

def Afqavoid (posion1,posion2,posion3,posion4):
    
    drones = np.array([
                       posion1,posion2,posion3,posion4
                       ])

    Afqformation = [0,0]
    Afqformation1 = [0,0]
    Afqformation2 = [0,0]
    Afqformation3 = [0,0]
    sigma4 = 1
    Ck = 0.1

    for i in range(len(drones)):
        if i != 0:
            # print(i)
            Afqformation_vetor =  posion1 - drones[i] 
            Afqformation_size = np.linalg.norm(Afqformation_vetor,ord=None)
        
            V_31 = Ck*1/Afqformation_size*np.exp(-np.square(Afqformation_size)/2*np.square(sigma4))*Afqformation_vetor
        
            Afqformation = Afqformation + V_31
        
    for i in range(len(drones)):
        if i != 1:
            # print(i)
            Afqformation_vetor1 =  posion2 - drones[i] 
            Afqformation_size1 = np.linalg.norm(Afqformation_vetor1,ord=None)
        
            V_32 = Ck*1/Afqformation_size1*np.exp(-np.square(Afqformation_size1)/2*np.square(sigma4))*Afqformation_vetor1
        
            Afqformation1 = Afqformation1 + V_32
    
    for i in range(len(drones)):
        if i != 2:
            Afqformation_vetor2 =  posion3 - drones[i] 
            Afqformation_size2 = np.linalg.norm(Afqformation_vetor2,ord=None)
        
            V_33 = Ck*1/Afqformation_size2*np.exp(-np.square(Afqformation_size2)/2*np.square(sigma4))*Afqformation_vetor2
        
            Afqformation2 = Afqformation2 + V_33

    for i in range(len(drones)):
        if i != 3:
            Afqformation_vetor3 =  posion4 - drones[i] 
            Afqformation_size3 = np.linalg.norm(Afqformation_vetor3,ord=None)
        
            V_34 = Ck*1/Afqformation_size3*np.exp(-np.square(Afqformation_size3)/2*np.square(sigma4))*Afqformation_vetor3
        
            Afqformation3 = Afqformation3 + V_34
    
    return -1*Afqformation,-1*Afqformation1,-1*Afqformation2,-1*Afqformation3
    # return [[0,0],[0,0],[0,0],[0,0]]

def Afqformation (r,r_d):
    r = np.array([r])
    r_d = np.array([r_d])
    A = 1
    V_1 = A*(r_d - r)
    return V_1
    # return [[0,0]]

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
    INIT_XYZS = np.array([[-3.0, 0.0, 0.5],[-5.0, 0.0, 0.5],[-4.0, 1.5, 0.5],[-4.0, -1.5, 0.5]])
    # print(INIT_XYZS[0, 2],INIT_XYZS[1, 2],'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1
    env = CtrlAviary(drone_model=drone,
                     num_drones=4,
                     initial_xyzs=INIT_XYZS,
                     physics=Physics.PYB_DW,
                     neighbourhood_radius=10,
                     freq=simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=gui,
                     record=record_video,
                     obstacles=True
                     )

    #### Initialize the trajectories ###########################
    PERIOD = 5
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP, 2))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = [0.5*np.cos(2*np.pi*(i/NUM_WP)), 0]
    
    wp_counters = np.array([0, int(NUM_WP/2)])

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=4,
                    duration_sec=duration_sec,
                    output_folder=output_folder,
                    colab=colab
                    )
    # p.loadURDF("cube_no_rotation.urdf", [0.5,-1.3,0], p.getQuaternionFromEuler([0,0,0]))#正方体
    # p.loadURDF("cube_no_rotation.urdf", [0.5,1.3,0], p.getQuaternionFromEuler([0,0,0]))#正方体
    #### Initialize the controllers ############################

    ctrl1 = [DSLPIDControl(drone_model=DroneModel('cf2x')) for i in range(1)]### 上方无人机的控制器
    ctrl2 = [DSLPIDControl(drone_model=DroneModel('cf2x')) for i in range(1)]### 下方无人机的控制器
    ctrl3 = [DSLPIDControl(drone_model=DroneModel('cf2x')) for i in range(1)]
    ctrl4 = [DSLPIDControl(drone_model=DroneModel('cf2x')) for i in range(1)]
    
    ##### Load the trained RL model ####################################
    model = PPO.load("ppo_cartpole4")
    obs = env.reset()
    act = {}

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(2)}
    START = time.time()
    j=0
    for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        
        for key, value in obs.items():
            act[key] = model.predict(obs['0']['state'])

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            ### 上方无人机的控制器 
            # INIT_XYZS = np.array([[-1, 0, 1],[-3, 0, 1],[-2.0, 1, 1],[-2.0, -1, 1]]) 
            j=j+1
            # print('jjjjjjjjjjjjjjjjjjjj:',i,CTRL_EVERY_N_STEPS)
            A = Afqavoid ([0,0],[-2,0],[-1,1.5],[-1,-1.5])
            A1 = [[0,0]]
            Afq1 = Afq([obs[str(0)][str("state")][0],obs[str(0)][str("state")][1]])
             
            
            p.resetDebugVisualizerCamera(
        cameraDistance=3,
        cameraYaw=-90,
        cameraPitch=-30,
        cameraTargetPosition=[-8+1*(i/240),0.0,2]
    )
            # log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robotmove.mp4")
            target_pos1 = [-3+1*(i/240),0.0,0.5]
            # target_pos1 = [obs[str(0)][str("state")][0],obs[str(0)][str("state")][1],0.8]
            action[str(0)], _, _ = ctrl1[0].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(0)]["state"],
                                                                target_pos=target_pos1,
                                                                target_vel=np.array([-Afq1[0]+A[0][0]+A1[0][0]+0.2,-Afq1[1]+A[0][1]+A1[0][1],0]),
                                                                ude=act['0'])    
            Afq2 = Afq([obs[str(1)][str("state")][0],obs[str(1)][str("state")][1]])
            A2 = Afqformation([obs[str(1)][str("state")][0],obs[str(1)][str("state")][1]],[obs[str(0)][str("state")][0]-2,obs[str(0)][str("state")][1]])
            target_pos2 = [-5+1*(i/240),0.0,0.5]
            # target_pos2 = [obs[str(1)][str("state")][0],obs[str(1)][str("state")][1],0.8]
            ### 下方无人机的控制器 
            action[str(1)], _, _ = ctrl2[0].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(1)]["state"],
                                                                target_pos=target_pos2,
                                                                target_vel=np.array([-Afq2[0]+A[1][0]+A2[0][0]+0.2,-Afq2[1]+A[1][1]+A2[0][1],0]),
                                                                ude=act['1']) 
            Afq3 = Afq([obs[str(2)][str("state")][0],obs[str(2)][str("state")][1]])
            A3 = Afqformation([obs[str(2)][str("state")][0],obs[str(2)][str("state")][1]],[obs[str(0)][str("state")][0]-1,obs[str(0)][str("state")][1]+1.5])
            target_pos3 = [-4+1*(i/240),1.5,0.5]
            # target_pos3 = [obs[str(2)][str("state")][0],obs[str(2)][str("state")][1],0.8]
            action[str(2)], _, _ = ctrl3[0].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(2)]["state"],
                                                                target_pos=target_pos3,
                                                                target_vel=np.array([-Afq3[0]+A[2][0]+A3[0][0]+0.2,-Afq3[1]+A[2][1]+A3[0][1],0]),
                                                                ude=act['2'])
            
            Afq4 = Afq([obs[str(3)][str("state")][0],obs[str(3)][str("state")][1]])
            A4 = Afqformation([obs[str(3)][str("state")][0],obs[str(3)][str("state")][1]],[obs[str(0)][str("state")][0]-1,obs[str(0)][str("state")][1]-1.5])
            target_pos4 = [-4+1*(i/240),-1.5,0.5]
            # target_pos4 = [obs[str(3)][str("state")][0],obs[str(3)][str("state")][1],0.8]
            action[str(3)], _, _ = ctrl4[0].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(3)]["state"],
                                                                target_pos=target_pos4,
                                                                target_vel=np.array([-Afq4[0]+A[3][0]+A4[0][0]+0.2,-Afq4[1]+A[3][1]+A4[0][1],0]),
                                                                ude=act['3'])
            
            # trajectory_des.append(np.array([target_pos1[0],target_pos1[1],target_pos2[0],target_pos2[1],target_pos3[0],target_pos3[1],target_pos4[0],target_pos4[1]]))
            trajectory_des.append([obs[str(1)][str("state")][0]+2,obs[str(1)][str("state")][1],
                                   obs[str(0)][str("state")][0]-2,obs[str(0)][str("state")][1],
                                   obs[str(0)][str("state")][0]-1,obs[str(0)][str("state")][1]+1.5,
                                   obs[str(0)][str("state")][0]-1,obs[str(0)][str("state")][1]-1.5,
                                   target_pos1[2],target_pos2[2],target_pos3[2],target_pos4[2]
                                   ])
            trajectory_real.append([obs[str(0)][str("state")][0],obs[str(0)][str("state")][1],
                                    obs[str(1)][str("state")][0],obs[str(1)][str("state")][1],
                                    obs[str(2)][str("state")][0],obs[str(2)][str("state")][1],
                                    obs[str(3)][str("state")][0],obs[str(3)][str("state")][1],

                                    obs[str(0)][str("state")][7],obs[str(0)][str("state")][8],obs[str(0)][str("state")][9],
                                    obs[str(1)][str("state")][7],obs[str(1)][str("state")][8],obs[str(1)][str("state")][9],
                                    obs[str(2)][str("state")][7],obs[str(2)][str("state")][8],obs[str(2)][str("state")][9],
                                    obs[str(3)][str("state")][7],obs[str(3)][str("state")][8],obs[str(3)][str("state")][9],
                                    obs[str(0)][str("state")][2],
                                    obs[str(1)][str("state")][2],
                                    obs[str(2)][str("state")][2],
                                    obs[str(3)][str("state")][2]
                                    ])
            
            # print('pppp:',trajectory_real[i],obs[str(0)][str("state")][1])
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

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    # p.stopStateLogging(log_id)
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("dw") # Optional CSV save

    #### Plot the simulation results ###########################
    # if plot:
    #     logger.plot()
    trajectory_des1 = np.array(trajectory_des)
    trajectory_real1 = np.array(trajectory_real)
    
    # np.savetxt('ccsicc/trajectory_des_pd.txt',trajectory_des1,fmt='%0.8f')
    # np.savetxt('ccsicc/trajectory_real_pd.txt',trajectory_real1,fmt='%0.8f')

    time1 = [0.01*i for i in range(len(trajectory_des))]
    plt.subplot(4,1,1)
    # plt.subplot(figsize=(12, 6))
    # plt.figure()
    plt.plot(trajectory_real1[:,0],trajectory_real1[:,1],color='b',linestyle='-.',linewidth=1,label='UAV1') 
    plt.plot(trajectory_real1[:,2],trajectory_real1[:,3],color='y',linestyle='-.',linewidth=1,label='UAV2')
    plt.plot(trajectory_real1[:,4],trajectory_real1[:,5],color='g',linestyle='-.',linewidth=1,label='UAV3')
    plt.plot(trajectory_real1[:,6],trajectory_real1[:,7],color='r',linestyle='-.',linewidth=1,label='UAV4')
    
    

    INIT_XYZ = np.array([[-3.0, 0.0, 0.8],[-4.0, 1.5, 0.8],[-5.0, 0.0, 0.8],[-4.0, -1.5, 0.8],[-3.0, 0.0, 0.8]])    
    # INIT_XYZZ = np.array([[-3.0, 0.0, 0.8],[-5.0, 0.0, 0.8],[-4.0, 1, 0.8],[-4.0, -1, 0.8]])
    k = 400
    j = len(trajectory_real1)-1
    XYZ =  np.array([[trajectory_real1[:,0][k],trajectory_real1[:,1][k]],[trajectory_real1[:,4][k],trajectory_real1[:,5][k]],[trajectory_real1[:,2][k],trajectory_real1[:,3][k]],[trajectory_real1[:,6][k],trajectory_real1[:,7][k]],[trajectory_real1[:,0][k],trajectory_real1[:,1][k]]])
    XYZ1 =  np.array([[trajectory_real1[:,0][j],trajectory_real1[:,1][j]],[trajectory_real1[:,4][j],trajectory_real1[:,5][j]],[trajectory_real1[:,2][j],trajectory_real1[:,3][j]],[trajectory_real1[:,6][j],trajectory_real1[:,7][j]],[trajectory_real1[:,0][j],trajectory_real1[:,1][j]]])
    # plt.figure(figsize=(14,7))
    plt.plot(INIT_XYZ[:,0],INIT_XYZ[:,1],color='gray',linestyle='-.',linewidth=1) 
    plt.plot(XYZ[:,0],XYZ[:,1],color='gray',linestyle='-.',linewidth=1)
    plt.plot(XYZ1[:,0],XYZ1[:,1],color='gray',linestyle='-.',linewidth=1)

    plt.scatter(INIT_XYZS[:,0][0],INIT_XYZS[:,1][0],color='b',s=50)
    plt.scatter(INIT_XYZS[:,0][1],INIT_XYZS[:,1][1],color='y',s=50)
    plt.scatter(INIT_XYZS[:,0][2],INIT_XYZS[:,1][2],color='g',s=50)
    plt.scatter(INIT_XYZS[:,0][3],INIT_XYZS[:,1][3],color='r',s=50)

    plt.scatter(XYZ[:,0][0],XYZ[:,1][0],color='b',s=50)
    plt.scatter(XYZ[:,0][1],XYZ[:,1][1],color='g',s=50)
    plt.scatter(XYZ[:,0][2],XYZ[:,1][2],color='y',s=50)
    plt.scatter(XYZ[:,0][3],XYZ[:,1][3],color='r',s=50)

    plt.scatter(XYZ1[:,0][0],XYZ1[:,1][0],color='b',s=50)
    plt.scatter(XYZ1[:,0][1],XYZ1[:,1][1],color='g',s=50)
    plt.scatter(XYZ1[:,0][2],XYZ1[:,1][2],color='y',s=50)
    plt.scatter(XYZ1[:,0][3],XYZ1[:,1][3],color='r',s=50)


    obstle = np.array([[-0,1.5],[-0,-1.5]])
    for i in np.arange(0,1,0.02):
        obstle=np.append(obstle, [[0,-2.5+i],[0,2.5-i]],axis = 0)

    for i in np.arange(0,2,0.02):
        obstle=np.append(obstle, [[i,1.5],[i,-1.5]],axis = 0)

    for i in np.arange(2,2.5,0.02):
        obstle=np.append(obstle, [[2,-i+0.5],[2,i-0.5]],axis = 0)

    for i in np.arange(2,6,0.02):
        obstle=np.append(obstle, [[i,2],[i,-2]],axis = 0)

    for i in np.arange(4,4.8,0.02):
        obstle=np.append(obstle, [[6,i-2.8],[6,-i+2.8]],axis = 0)

    for i in np.arange(6,10,0.02):
        obstle=np.append(obstle, [[i,1.2],[i,-1.2]],axis = 0)

   
    for f in range(len(obstle)):
        plt.scatter(obstle[:,0][f],obstle[:,1][f],color='black',)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid(alpha=0.1)
  
    ### x_erro
    # plt.subplot(3, 2, 1)
    # plt.plot(time1,trajectory_real1[:,0]-trajectory_des1[:,0],color='b',linestyle='-.',linewidth=1,label='UAV1') 
    # plt.plot(time1,trajectory_real1[:,2]-trajectory_des1[:,2],color='y',linestyle='-.',linewidth=1,label='UAV2')
    # plt.plot(time1,trajectory_real1[:,4]-trajectory_des1[:,4],color='g',linestyle='-.',linewidth=1,label='UAV3')
    # plt.plot(time1,trajectory_real1[:,6]-trajectory_des1[:,6],color='r',linestyle='-.',linewidth=1,label='UAV4')
    # plt.xlabel('time[s]')
    # plt.ylabel('x_erro [m]')

    # plt.xlim((0,0.01*len(time1)))
    # plt.ylim((-1.2,1.2))
    # plt.legend(loc = 'upper left')
    # plt.grid(alpha=0.1)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)  
    # # plt.xlabel(fontsize=20)
    # # plt.ylabel( fontsize=20) 

    # ## y_erro
    # plt.subplot(3, 2, 2)
    # plt.plot(time1,trajectory_real1[:,1]-trajectory_des1[:,1],color='b',linestyle='-.',linewidth=1,label='UAV1') 
    # plt.plot(time1,trajectory_real1[:,3]-trajectory_des1[:,3],color='y',linestyle='-.',linewidth=1,label='UAV2')
    # plt.plot(time1,trajectory_real1[:,5]-trajectory_des1[:,5],color='g',linestyle='-.',linewidth=1,label='UAV3')
    # plt.plot(time1,trajectory_real1[:,7]-trajectory_des1[:,7],color='r',linestyle='-.',linewidth=1,label='UAV4')
    # plt.xlabel('time [s]')
    # plt.ylabel('y_erro [m]')
    # plt.xlim((0,0.01*len(time1)))
    # plt.ylim((-1.2,1.2))
    # # plt.legend(loc = 'upper left')
    # plt.grid(alpha=0.1)

    # ## z_erro
    # plt.subplot(3, 2, 3)
    # plt.plot(time1,trajectory_real1[:,20]-trajectory_des1[:,8],color='b',linestyle='-.',linewidth=1,label='UAV1') 
    # plt.plot(time1,trajectory_real1[:,21]-trajectory_des1[:,9],color='y',linestyle='-.',linewidth=1,label='UAV2')
    # plt.plot(time1,trajectory_real1[:,22]-trajectory_des1[:,10],color='g',linestyle='-.',linewidth=1,label='UAV3')
    # plt.plot(time1,trajectory_real1[:,23]-trajectory_des1[:,11],color='r',linestyle='-.',linewidth=1,label='UAV4')
    # plt.xlabel('time[s]')
    # plt.ylabel('z_erro [m]')
    # plt.xlim((0,0.01*len(time1)))
    # plt.grid(alpha=0.1)
    # # plt.legend(loc = 'upper left')

    # # plt.axes([0.6,0.1,0.2,0.2])  
    # # plt.plot(time1[200:],trajectory_real1[200:,20]-trajectory_des1[200:,8],'y')
    # # plt.title('little axes2')
    # # # plt.ylim((-1,1))
    # # # plt.legend()
    # # plt.grid(alpha=0.1)
    

    # plt.subplot(3, 2, 4)
    # plt.plot(time1,trajectory_real1[:,8],color='b',linestyle='-.',linewidth=1,label='UAV1') 
    # plt.plot(time1,trajectory_real1[:,11],color='y',linestyle='-.',linewidth=1,label='UAV2')
    # plt.plot(time1,trajectory_real1[:,14],color='g',linestyle='-.',linewidth=1,label='UAV3')
    # plt.plot(time1,trajectory_real1[:,17],color='r',linestyle='-.',linewidth=1,label='UAV4')
    # plt.xlabel('time[s]')
    # plt.ylabel('roll [degree]')
    # plt.xlim((0,0.01*len(time1)))
    # plt.ylim((-0.15,0.15))
    # # plt.legend(loc = 'upper left')
    # plt.grid(alpha=0.1)

    # plt.subplot(3, 2, 5)
    # plt.plot(time1,trajectory_real1[:,9],color='b',linestyle='-.',linewidth=1,label='UAV1') 
    # plt.plot(time1,trajectory_real1[:,12],color='y',linestyle='-.',linewidth=1,label='UAV2')
    # plt.plot(time1,trajectory_real1[:,15],color='g',linestyle='-.',linewidth=1,label='UAV3')
    # plt.plot(time1,trajectory_real1[:,18],color='r',linestyle='-.',linewidth=1,label='UAV4')
    # plt.xlabel('time[s]')
    # plt.ylabel('pitch [degree]')
    # plt.xlim((0,0.01*len(time1)))
    # plt.ylim((-0.15,0.15))
    # # plt.legend(loc = 'upper left')
    # plt.grid(alpha=0.1)

    # plt.subplot(3, 2, 6)
    # plt.plot(time1,trajectory_real1[:,10],color='b',linestyle='-.',linewidth=1,label='UAV1') 
    # plt.plot(time1,trajectory_real1[:,13],color='y',linestyle='-.',linewidth=1,label='UAV2')
    # plt.plot(time1,trajectory_real1[:,16],color='g',linestyle='-.',linewidth=1,label='UAV3')
    # plt.plot(time1,trajectory_real1[:,19],color='r',linestyle='-.',linewidth=1,label='UAV4')
    # plt.xlabel('time[s]')
    # plt.ylabel('yaw [degree]')
    # plt.xlim((0,0.01*len(time1)))
    # plt.ylim((-0.15,0.15))
    # # plt.legend(loc = 'upper left')
    # plt.grid(alpha=0.1)


    plt.show()
    plt.savefig('formation.pdf',dpi =350)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Downwash example script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--aggregate',          default=DEFAULT_AGGREGATE,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()
 
    
    run(**vars(ARGS))