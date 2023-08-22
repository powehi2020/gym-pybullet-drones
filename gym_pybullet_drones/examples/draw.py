import numpy as np
trajectory_des1 = np.loadtxt('/home/lkd/Documents/GitHub/gym-pybullet-drones/gym_pybullet_drones/examples/ccsicc/trajectory_des_pd.txt')#最普通的loadtxt
trajectory_real1 = np.loadtxt('/home/lkd/Documents/GitHub/gym-pybullet-drones/gym_pybullet_drones/examples/ccsicc/trajectory_real_pd.txt')
import matplotlib.pyplot as plt
time1 = [0.01*i for i in range(len(trajectory_des1))]
plt.figure(figsize=(24,18))
plt.subplot(2,1,1)
# plt.subplot(figsize=(24, 18))


# plt.figure()
plt.plot(trajectory_real1[:,0],trajectory_real1[:,1],color='#ff2d37',linestyle='-.',linewidth=1.5,label='UAV1') 
plt.plot(trajectory_real1[:,2],trajectory_real1[:,3],color='#2db928',linestyle='-.',linewidth=1.5,label='UAV2')
plt.plot(trajectory_real1[:,4],trajectory_real1[:,5],color='#ffa500',linestyle='-.',linewidth=1.5,label='UAV3')
plt.plot(trajectory_real1[:,6],trajectory_real1[:,7],color='#004eaf',linestyle='-.',linewidth=1.5,label='UAV4')



INIT_XYZ = np.array([[-3.0, 0.0, 0.8],[-4.0, 1.5, 0.8],[-5.0, 0.0, 0.8],[-4.0, -1.5, 0.8],[-3.0, 0.0, 0.8]])    
# INIT_XYZZ = np.array([[-3.0, 0.0, 0.8],[-5.0, 0.0, 0.8],[-4.0, 1, 0.8],[-4.0, -1, 0.8]])
# pd
k = 220
f = 400
l = 700

# pd+ude
# k = 350
# f = 550
# l = 755


j = len(trajectory_real1)-1
XYZ =  np.array([[trajectory_real1[:,0][k],trajectory_real1[:,1][k]],[trajectory_real1[:,4][k],trajectory_real1[:,5][k]],[trajectory_real1[:,2][k],trajectory_real1[:,3][k]],[trajectory_real1[:,6][k],trajectory_real1[:,7][k]],[trajectory_real1[:,0][k],trajectory_real1[:,1][k]]])
XYZ1 =  np.array([[trajectory_real1[:,0][j],trajectory_real1[:,1][j]],[trajectory_real1[:,4][j],trajectory_real1[:,5][j]],[trajectory_real1[:,2][j],trajectory_real1[:,3][j]],[trajectory_real1[:,6][j],trajectory_real1[:,7][j]],[trajectory_real1[:,0][j],trajectory_real1[:,1][j]]])

XYZ2 =  np.array([[trajectory_real1[:,0][f],trajectory_real1[:,1][f]],
                  [trajectory_real1[:,4][f],trajectory_real1[:,5][f]],
                  [trajectory_real1[:,2][f],trajectory_real1[:,3][f]],
                  [trajectory_real1[:,6][f],trajectory_real1[:,7][f]],
                  [trajectory_real1[:,0][f],trajectory_real1[:,1][f]]])

XYZ3=  np.array([[trajectory_real1[:,0][l],trajectory_real1[:,1][l]],
                  [trajectory_real1[:,4][l],trajectory_real1[:,5][l]],
                  [trajectory_real1[:,2][l],trajectory_real1[:,3][l]],
                  [trajectory_real1[:,6][l],trajectory_real1[:,7][l]],
                  [trajectory_real1[:,0][l],trajectory_real1[:,1][l]]])

INIT_XYZS = np.array([[-3.0, 0.0, 0.5],[-5.0, 0.0, 0.5],[-4.0, 1.5, 0.5],[-4.0, -1.5, 0.5]])

# plt.figure(figsize=(14,7))
plt.plot(INIT_XYZ[:,0],INIT_XYZ[:,1],color='gray',linestyle='-.',linewidth=1) 
plt.plot(XYZ[:,0],XYZ[:,1],color='gray',linestyle='-.',linewidth=1)
plt.plot(XYZ1[:,0],XYZ1[:,1],color='gray',linestyle='-.',linewidth=1)
plt.plot(XYZ2[:,0],XYZ2[:,1],color='gray',linestyle='-.',linewidth=1)
plt.plot(XYZ3[:,0],XYZ3[:,1],color='gray',linestyle='-.',linewidth=1)

plt.scatter(INIT_XYZS[:,0][0],INIT_XYZS[:,1][0],color='#ff2d37',s=50)
plt.scatter(INIT_XYZS[:,0][1],INIT_XYZS[:,1][1],color='#2db928',s=50)
plt.scatter(INIT_XYZS[:,0][2],INIT_XYZS[:,1][2],color='#ffa500',s=50)
plt.scatter(INIT_XYZS[:,0][3],INIT_XYZS[:,1][3],color='#004eaf',s=50)

plt.scatter(XYZ[:,0][0],XYZ[:,1][0],color='#ff2d37',s=50)
plt.scatter(XYZ[:,0][1],XYZ[:,1][1],color='#ffa500',s=50)
plt.scatter(XYZ[:,0][2],XYZ[:,1][2],color='#2db928',s=50)
plt.scatter(XYZ[:,0][3],XYZ[:,1][3],color='#004eaf',s=50)

plt.scatter(XYZ1[:,0][0],XYZ1[:,1][0],color='#ff2d37',s=50)
plt.scatter(XYZ1[:,0][1],XYZ1[:,1][1],color='#ffa500',s=50)
plt.scatter(XYZ1[:,0][2],XYZ1[:,1][2],color='#2db928',s=50)
plt.scatter(XYZ1[:,0][3],XYZ1[:,1][3],color='#004eaf',s=50)

plt.scatter(XYZ2[:,0][0],XYZ2[:,1][0],color='#ff2d37',s=50)
plt.scatter(XYZ2[:,0][1],XYZ2[:,1][1],color='#ffa500',s=50)
plt.scatter(XYZ2[:,0][2],XYZ2[:,1][2],color='#2db928',s=50)
plt.scatter(XYZ2[:,0][3],XYZ2[:,1][3],color='#004eaf',s=50)

plt.scatter(XYZ3[:,0][0],XYZ3[:,1][0],color='#ff2d37',s=50)
plt.scatter(XYZ3[:,0][1],XYZ3[:,1][1],color='#ffa500',s=50)
plt.scatter(XYZ3[:,0][2],XYZ3[:,1][2],color='#2db928',s=50)
plt.scatter(XYZ3[:,0][3],XYZ3[:,1][3],color='#004eaf',s=50)


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

plt.xlabel('x [m]',fontsize=15)
plt.ylabel('y [m]',fontsize=15)
plt.ylim((-2.5,2.5))
plt.xlim((-6,15.5))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10) 
plt.legend(loc = 'upper left')
plt.grid(alpha=0.1)


# # 设置刻度字体大小
# plt.xticks(fontsize=5)
# plt.yticks(fontsize=5)
# 设置坐标标签字体大小


plt.savefig('/home/lkd/Documents/GitHub/gym-pybullet-drones/gym_pybullet_drones/examples/ccsicc/pd_formation.png',dpi=200)
plt.savefig('/home/lkd/Documents/GitHub/gym-pybullet-drones/gym_pybullet_drones/examples/ccsicc/pd_formation.pdf',dpi=200)
plt.show()