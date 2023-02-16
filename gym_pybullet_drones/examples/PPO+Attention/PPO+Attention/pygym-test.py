from ruamel.yaml import YAML, dump, RoundTripDumper
import os,time,subprocess
import numpy as np
from flightgym import QuadrotorEnv_v1
import platform


cfg = YAML().load(open("flightenv.yaml", 'r'))

cfg["env"]["render"] = "yes"

plat = platform.system()
if plat == "Windows":
    unity = subprocess.Popen(os.environ["FLIGHTMARE_PATH"] +"/flightrender/RPG_Flightmare.exe")
elif plat == "Linux":
    unity = subprocess.Popen(os.environ["FLIGHTMARE_PATH"] +"/flightrender/RPG_Flightmare.X86_64")
else:
    print("platform not supported")
    exit()

env = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)

env.connectUnity()

state = np.zeros((2,13), np.float32)
state[0,3] = 1
state[1,3] = 1

state[0,0] = 0
state[0,2] = 0.2
state[1,0] = 1
state[1,2] = 0.2

frame = 0
t0 = time.time()
rate = 60
v = 0.5
while frame < rate*6:
    env.setState(state, frame)
    state[0,1] += v / rate
    state[1,1] += v / rate
    # print(state[0,1])
    t1 = time.time()
    if t1 < t0 + 1/rate:
        time.sleep(t0 + 1/rate - t1)
    t0 = t1
    frame = frame + 1

while frame < rate*12:
    env.setState(state, frame)
    state[0,0] += v / rate
    state[1,0] += v / rate
    # print(state[0,1])
    t1 = time.time()
    if t1 < t0 + 1/rate:
        time.sleep(t0 + 1/rate - t1)
    t0 = t1
    frame = frame + 1

exit()