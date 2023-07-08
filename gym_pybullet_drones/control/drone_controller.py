from . import pid_controller as pid
from . import drone_config as config
import math

dt = config.dt

class drone_controller:
    def __init__(self) -> None:
        self.__x_pid = pid.pid_controller(config.PARAMS_POS_X_PID.KP,
        config.PARAMS_POS_X_PID.KI,
        config.PARAMS_POS_X_PID.KD,
        config.PARAMS_POS_X_PID.CD,
        dt,-config.PARAMS_PID_LIMIT.VELX,config.PARAMS_PID_LIMIT.VELX)
        
        self.__y_pid = pid.pid_controller(config.PARAMS_POS_Y_PID.KP,
        config.PARAMS_POS_Y_PID.KI,
        config.PARAMS_POS_Y_PID.KD,
        config.PARAMS_POS_Y_PID.CD,
        dt,-config.PARAMS_PID_LIMIT.VELY,config.PARAMS_PID_LIMIT.VELY)

        self.__z_pid = pid.pid_controller(config.PARAMS_POS_Z_PID.KP,
        config.PARAMS_POS_Z_PID.KI,
        config.PARAMS_POS_Z_PID.KD,
        config.PARAMS_POS_Z_PID.CD,
        dt,-config.PARAMS_PID_LIMIT.VELZ,config.PARAMS_PID_LIMIT.VELZ)

        self.__vx_pid = pid.pid_controller(config.PARAMS_VEL_X_PID.KP,
        config.PARAMS_VEL_X_PID.KI,
        config.PARAMS_VEL_X_PID.KD,
        config.PARAMS_VEL_X_PID.CD,
        dt,-config.PARAMS_PID_LIMIT.PITCH,config.PARAMS_PID_LIMIT.PITCH)

        self.__vy_pid = pid.pid_controller(config.PARAMS_VEL_Y_PID.KP,
        config.PARAMS_VEL_Y_PID.KI,
        config.PARAMS_VEL_Y_PID.KD,
        config.PARAMS_VEL_Y_PID.CD,
        dt,-config.PARAMS_PID_LIMIT.ROLL,config.PARAMS_PID_LIMIT.ROLL)
        
        vmax = (config.PARAMS_Thrust.MAX - config.PARAMS_Thrust.BASE)/config.PARAMS_Thrust.SCALE
        vmin = -(config.PARAMS_Thrust.MAX - config.PARAMS_Thrust.BASE)/config.PARAMS_Thrust.SCALE
        self.__vz_pid = pid.pid_controller(config.PARAMS_VEL_Z_PID.KP,
        config.PARAMS_VEL_Z_PID.KI,
        config.PARAMS_VEL_Z_PID.KD,
        config.PARAMS_VEL_Z_PID.CD,
        dt,vmin,vmax)

        self.__yaw_pid = pid.pid_controller(config.PARAMS_YAW_PID.KP,
        config.PARAMS_YAW_PID.KI,
        config.PARAMS_YAW_PID.KD,
        config.PARAMS_YAW_PID.CD,
        dt)
        # 
        self.__mode_pos_x = True
        self.__mode_pos_y = True
        self.__mode_pos_z = True
        self.__mode_yaw_or_rate  = False   
        self.__mode_vel_ff_x = False
        self.__mode_vel_ff_y = False
        self.__mode_vel_ff_z = False

    def controller_update(self,pos,pos_d,vel,vel_d,yaw):
        # Global
        x = pos[0]
        y = pos[1]
        z = pos[2]
        x_d = pos_d[0]
        y_d = pos_d[1]
        z_d = pos_d[2]
        # Body 
        cosyaw = math.cos(yaw*math.pi/180.0)
        sinyaw = math.sin(yaw*math.pi/180.0)

        # xr =  x * cosyaw + y * sinyaw
        # yr = -x * sinyaw + y * cosyaw
        # xr_d =  x_d * cosyaw + y_d * sinyaw
        # yr_d = -x_d * sinyaw + y_d * cosyaw

        xr =  x 
        yr =  y 
        xr_d =  x_d 
        yr_d =  y_d 
        
        # Position Control
        vx_u = 0
        # if self.__mode_pos_x:
        vx_u = self.__x_pid.update(xr_d - xr)
        

        vy_u = 0
        # if self.__mode_pos_y:
        vy_u = self.__y_pid.update(yr_d - yr)
    

        vz_u = 0
        # if self.__mode_pos_z:
        vz_u = self.__z_pid.update(z_d - z)

        # Global
        vx = vel[0]
        vy = vel[1]
        vz = vel[2]
        # Body 
        # vxr =  vx * cosyaw + vy * sinyaw
        # vyr = -vx * sinyaw + vy * cosyaw

        vxr =  vx 
        vyr =  vy 


        # Velocity feedforward
        # if self.__mode_vel_ff_x:
        # vxr_d =   vel_d[0] * cosyaw + vel_d[1] * sinyaw
        vxr_d =   vel_d[0] 


        vx_u = vxr_d
        # if self.__mode_vel_ff_y:
        # vyr_d =  -vel_d[0] * sinyaw + vel_d[1] * cosyaw

        vyr_d =  vel_d[1] 

        vy_u = vyr_d
        # if self.__mode_vel_ff_z:


       
        
        # Velocity Control
        pitch_u = self.__vx_pid.update(vx_u - vxr)
        roll_u = -self.__vy_pid.update(vy_u - vyr)
        # thrust_u = self.__vz_pid.update(vz_u - vz) * config.PARAMS_Thrust.SCALE + config.PARAMS_Thrust.BASE

        thrust_u = self.__vz_pid.update(vz_u - vz) 
        

        if thrust_u < config.PARAMS_Thrust.MIN:
            thrust_u = config.PARAMS_Thrust.MIN

        # Yaw Control
       
        yaw_rate_u = 0

        return roll_u,pitch_u,yaw_rate_u,thrust_u
        
    def reset_controller(self):
        self.__x_pid.reset_former()
        self.__y_pid.reset_former()
        self.__z_pid.reset_former()
        self.__vx_pid.reset_former()
        self.__vy_pid.reset_former()
        self.__vz_pid.reset_former()
        self.__yaw_pid.reset_former()
    def reset_controller_integral(self):
        self.__x_pid.reset_integral()
        self.__y_pid.reset_integral()
        self.__z_pid.reset_integral()
        self.__vx_pid.reset_integral()
        self.__vy_pid.reset_integral()
        self.__vz_pid.reset_integral()
        self.__yaw_pid.reset_integral()

    def set_mode(self,mode):
        # 0: pos, yaw, no vel
        # 1: z, no horizontal pos, horizontal vel, no z vel, yaw_rate
        if mode == 0:
            self.__mode_pos_x = True
            self.__mode_pos_y = True
            self.__mode_pos_z = True
            self.__mode_yaw_or_rate = True
            self.__mode_vel_ff_x = False
            self.__mode_vel_ff_y = False
            self.__mode_vel_ff_z = False
        elif mode == 1:
            self.__mode_pos_x = False
            self.__mode_pos_y = False
            self.__mode_pos_z = True
            self.__mode_yaw_or_rate = False
            self.__mode_vel_ff_x = True
            self.__mode_vel_ff_y = True
            self.__mode_vel_ff_z = False
        else:
            self.__mode_pos_x = True
            self.__mode_pos_y = True
            self.__mode_pos_z = True
            self.__mode_yaw_or_rate = True
            self.__mode_vel_ff_x = False
            self.__mode_vel_ff_y = False
            self.__mode_vel_ff_z = False

