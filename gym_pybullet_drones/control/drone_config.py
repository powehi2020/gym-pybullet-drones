# crazyflie

dt = 1.0/120.0
class PARAMS_POS_X_PID:
    KP = 100.0
    KI = 0.0
    KD = 0.0
    CD = 0.0
    
class PARAMS_POS_Y_PID:
    KP = 1.0
    KI = 0.0
    KD = 0.0
    CD = 0.0

class PARAMS_POS_Z_PID:
    KP = 0.0
    KI = 0.0
    KD = 0.0
    CD = 0.0

class PARAMS_VEL_X_PID:
    KP = 0.1
    KI = 0.0
    KD = 0.0
    CD = 0.0

class PARAMS_VEL_Y_PID:
    KP = 1.0
    KI = 0.0
    KD = 0.0
    CD = 0.0

class PARAMS_VEL_Z_PID:
    KP = 0.0
    KI = 0.0
    KD = 0.0
    CD = 0.0

class PARAMS_YAW_PID:
    KP = 6.0
    KI = 0.0
    KD = 0.35
    CD = 1.0
    I_LIMIT = 360.0

class PARAMS_PID_LIMIT:
    VELX = 1.0
    VELY = 1.0
    VELZ = 0.5
    VELH = 1.0  # horizontal plane
    ROLL = 20.0 # deg
    PITCH = 20.0 # deg 

class PARAMS_Thrust:
    BASE = 42000.0
    SCALE = 1000.0
    MIN = 20000 # 10001
    MAX = 46000 # 60000

