import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
import random

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.control.drone_controller import drone_controller
from gym_pybullet_drones.utils.enums import DroneModel
import time

'''
ude control
'''

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu, 
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])

        # self.P_COEFF_TOR = np.array([3000., 3000., 3000.])
        # self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.I_COEFF_TOR = np.array([.0, .0, 0])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        # self.D_COEFF_TOR = np.array([300., 300., 300.])
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.acc_x = 0
        self.acc_y = 0
        self.acc_z = 0
        self.torque_x=0
        self.torque_y=0
        self.torque_z=0

        self.rpq = np.array([.0, .0, 0])

        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([ [0, -1,  -1], [+1, 0, 1], [0,  1,  -1], [-1, 0, 1] ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.last_vel_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        # rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        ####ude
        self.integral_rpy = np.zeros(3)
        self.integral_u = np.zeros(3)

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        k_p = [15,15,30]
        k_d = [10,10,12]
        # target_vel = np.array([0., 0., 0.])
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        vel_e = np.clip(vel_e,-2,2)

        self.reward_pos = pos_e[2]
        self.reward_vel = vel_e[2]
        self.done_pos = cur_pos

        # print(cur_pos,'kk')

        self.pos = cur_pos
        print(cur_pos)
        self.vel = vel_e

        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)

        #### UDE target thrust #####################################

        # T_ude = A[0]
        # T_ude = 0.7
        #### 位置控制 #####################################
        vx = k_p[0]*pos_e[0] + k_d[0]*vel_e[0] 
        vy = k_p[1]*pos_e[1] + k_d[1]*vel_e[1]
        vz = k_p[2]*pos_e[2] + k_d[2]*vel_e[2]


        acc_0 = k_p[0]*pos_e[0] + k_d[0]*vel_e[0] 
        acc_1 = k_p[1]*pos_e[1] + k_d[1]*vel_e[1]
        acc_2 = k_p[2]*pos_e[2] + k_d[2]*vel_e[2]

        
        
        #### 速度控制 #####################################
        kv_p = [1,1,2]
        kv_d = [0,0,0]
        v_d = [1/9.8*vx,1/9.8*vy,vz]

        v_e = v_d - cur_vel 

        

        v_de = (v_e - self.last_vel_e)/control_timestep

        print(v_e,v_de)

        u_roll = kv_p[0]*(v_e[0]) +  kv_d[0]*v_de[0]
        u_pitch = (kv_p[1]*(v_e[1]) +  kv_d[0]*v_de[1])
        thrust1 = kv_p[2]*(v_e[2]) +  kv_d[0]*(v_de[2])
       
        acc_0 = np.clip(acc_0,-2,2)
        acc_1 = np.clip(acc_1,-2,2)

        #### UDE 控制 #####################################
        # self.acc_x = self.acc_x + acc_0*control_timestep
        # self.acc_y = self.acc_y + acc_1*control_timestep
        # self.acc_z = self.acc_z + acc_2*control_timestep

        # f_x = - 1/T_ude *(self.acc_x-cur_vel[0])
        # f_y = - 1/T_ude *(self.acc_y-cur_vel[1])
        # f_z = - 1/T_ude *(self.acc_z-cur_vel[2])

        f_x = 0
        f_y = 0
        f_z = 0

        
        thrust1 = self.GRAVITY + self.GRAVITY/9.8*(thrust1)
        # thrust1 = self.GRAVITY + self.GRAVITY/9.8*(acc_2 - f_z)
        thrust = (math.sqrt(thrust1 / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        phi_des_dd = 1/9.8*(-acc_1+f_y)  
        theta_des_dd = 1/9.8*(acc_0-f_x) 

        # target_euler = np.array([phi_des_dd,theta_des_dd,0. ])
        target_euler = np.array([-u_pitch,u_roll,0. ])


        self.last_pos_e = [pos_e[0],pos_e[1],pos_e[2]]
        self.last_vel_e = v_e
        


        ## 存取数据/home/lkd/test_data/2023_3_14
        # with open ('/home/lkd/test_data/'+time.strftime("%Y_%m_%d", time.localtime())+'/x_c.txt','a') as f:
        #     f.write(str(cur_pos[0]))
        #     f.write('\n')
        


        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
            
            
        
        
            
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        # cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))

        self.cur_rpy = cur_rpy
        # target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        # w,x,y,z = target_quat
        # target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        # rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        # rot_e = -np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 

        rot_e = target_euler- cur_rpy      
        self.rpy = rot_e
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        self.reward_rpy = rpy_rates_e
        cur_rpy_rates = (cur_rpy - self.last_rpy)/control_timestep

        I = np.array([[1.43e-5,0,0],[0,1.43e-5,0],[0,0,2.89e-5]])
        kp_m = np.array([3000,3000,3000])
        kd_m = np.array([300,300,300])

        #### PD target torques ####################################
        
        torque_x=kp_m[0]*rot_e[0] + kd_m[0]*rpy_rates_e[0]
        torque_y=kp_m[1]*rot_e[1] + kd_m[1]*rpy_rates_e[1]
        torque_z=kp_m[2]*rot_e[2] + kd_m[2]*rpy_rates_e[2]

        #### UDE design ####################################
        # A = self.get_action()

        # T_torque_ude = A[0]
        # print(A[0])

        self.torque_x = self.torque_x + torque_x*control_timestep
        self.torque_y = self.torque_y + torque_y*control_timestep
        self.torque_z = self.torque_z + torque_z *control_timestep
        
        # f_torque_x = 1 / T_torque_ude *(cur_rpy_rates[0]-self.torque_x)
        # f_torque_y = 1 / T_torque_ude *(cur_rpy_rates[1]-self.torque_y)
        # f_torque_z = 1 / T_torque_ude *(cur_rpy_rates[2]-self.torque_z)
    
        f_torque_x = 0
        f_torque_y = 0
        f_torque_z = 0


        target_torques = np.array([torque_x-f_torque_x ,torque_y-f_torque_y,torque_z-f_torque_z])
       
                  
        target_torques = np.dot(I,target_torques) /self.KM
        # print(target_torques,'target_torques')
        target_torques = np.clip(target_torques, -3200, 3200)



        # pwm is the motor control signal 
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    
    def compute_reward(self):
        '''compute the reward of the current state'''                 
        c_p = 4e-3
        c_v = 5e-4
        c_rpy =34e-1
        c = 0
        if self.compute_done == True:
            c=1
        reward = - (c_p * np.linalg.norm(self.reward_pos) + c_v * np.linalg.norm(self.reward_vel)+ c_rpy * np.linalg.norm(self.rpy)+ c_rpy * np.linalg.norm((self.reward_rpy)+c))
        print(reward,'reward')
        return reward
    ################################################################################

    def compute_done(self):
        '''compute the reward of the current state'''   
                      
        
        if   self.pos[1]<-0.45 or self.pos[1]>0.45 or self.pos[2]<0.1:
            # print(self.cur_rpy[1])
            print('done')
            # print(self.pos[0],'aa')
            # print(self.pos[1])
            # print(self.vel[2])
            return True
        else:
            print(self.pos[0],'aa')
            print(self.pos[1])
            print(self.pos[2])
            return False
        
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
