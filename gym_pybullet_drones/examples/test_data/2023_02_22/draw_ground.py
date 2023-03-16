#################################################################
#对分别包含xyz，位置信息的txt文件进行画图分析
#
#################################################################

import matplotlib.pyplot as plt
import numpy as np
for j in ['x','y','z']:
    
    X, Y, Z, N, M, C, A, B = [], [], [], [], [], [], [], []
    star = []
    for line in open('groud_pd/'+j+'_c.txt', 'r'):
        values = [float(s) for s in line.split()]
        X.append(values[0])

        
    for i in range (len(X)):
        print(len(X))
        Y.append(0.01*i)
     
    for line in open('groud_pd/'+j+'_t.txt', 'r'):
        values = [float(s) for s in line.split()]
        Z.append(values[0])

    for line in open('groud_ude0.5/'+j+'_c.txt', 'r'):
        values = [float(s) for s in line.split()]
        N.append(values[0])
                    
    for line in open('groud_ude1/'+j+'_c.txt', 'r'):
        values = [float(s) for s in line.split()]
        M.append(values[0])

    star = np.array([X[0]])
    ref = np.array([0,0,0.5])
    
    fig, ax = plt.subplots(figsize=(5, 5))

    if j == "x":
        ax.set_ylim(-1.2,1.2)
    else:
        ax.set_ylim(-0.5,0.5)


    A = [X[i]-Z[i] for i in range(0,len(X))]
    B = [N[i]-Z[i] for i in range(0,len(N))]
    C = [M[i]-Z[i] for i in range(0,len(M))]
    D = [Z[i]-Z[i] for i in range(0,len(Z))]


    ### real state
    ax.plot(Y, X, label='PD',color = 'r')  # Plot some data on the axes.
    ax.plot(Y, N, label='UDE=0.5',color = 'g')  # Plot some data on the axes.
    ax.plot(Y, M, label='UDE=1',color = 'y')  # Plot some data on the axes.
    ax.plot(Y, Z, label='target',color = 'b')  # Plot more data on the axes...
    plt.scatter(Y[0], star[0], label='start',color = 'k')
    
    #### Erro 
    # ax.plot(Y, A, label='PD_erro',color = 'r')  # Plot some data on the axes.
    # ax.plot(Y, B, label='UDE=0.5',color = 'g')  # Plot some data on the axes.
    # ax.plot(Y, C, label='UDE=1',color = 'y')  # Plot some data on the axes.
    # ax.plot(Y, D, label='target',color = 'b')  # Plot more data on the axes...
    


    
    plt.ylabel(j+' (m)')  #横坐标含义
    plt.xlabel('time(s)')  #纵坐标含义
    plt.title(j+' direction')
    plt.legend()
    plt.savefig(j+'.png',dpi=600,bbox_inches="tight")
    plt.show()
