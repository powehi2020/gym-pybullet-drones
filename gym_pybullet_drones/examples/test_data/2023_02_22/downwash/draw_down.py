#################################################################
#对分别包含xyz，位置信息的txt文件进行画图分析
#
#################################################################

import matplotlib.pyplot as plt
import numpy as np
for j in ['x','y','z']:
    
    X, Y, Z, N , M ,T ,U, A, B, C, D = [], [], [], [], [], [], [], [], [], [], []
    star = []
    for line in open('downwash_pd/'+j+'_t.txt', 'r'):
        values = [float(s) for s in line.split()]
        X.append(values[0])
     
    for line in open('downwash_pd/'+j+'_c.txt', 'r'):
        values = [float(s) for s in line.split()]
        Z.append(values[0])

    for line in open('downwash_ude0.5/'+j+'_c.txt', 'r'):
        values = [float(s) for s in line.split()]
        N.append(values[0])
                    
    for line in open('downwash_ude1/'+j+'_c.txt', 'r'):
        values = [float(s) for s in line.split()]
        M.append(values[0])

    for i in range (len(Z)):
        # print(len(X))
        Y.append(0.01*i)

    for i in range (len(N)):
        # print(len(X))
        T.append(0.01*i)

    for i in range (len(M)):
        print(len(X))
        U.append(0.01*i)

    star = np.array([X[0]])
    # ref = np.array([0,0,0.5])

    A = [Z[i]-X[i] for i in range(0,len(X))]
    B = [N[i]-X[i] for i in range(0,len(N))]
    C = [M[i]-X[i] for i in range(0,len(M))]
    D = [X[i]-X[i] for i in range(0,len(Z))]
    
    fig, ax = plt.subplots(figsize=(5, 5))

    if j == "x":
        ax.set_ylim(-1.2,1.2)
    else:
       ax.set_ylim(-0.2,1)
      
    ax.plot(Y, Z, label='pd',color = 'r')  # Plot some data on the axes.
    ax.plot(T, N, label='ude=0.5',color = 'k')  # Plot some data on the axes.  
    ax.plot(U, M, label='ude=1',color = 'y')  # Plot some data on the axes.
    ax.plot(Y, X, label='target',color = 'b')  # Plot more data on the axes...
    plt.scatter(Y[0], star[0], label='start',color = 'k')

    ### erro 
    # ax.plot(Y, A, label='pd_erro',color = 'r')  # Plot some data on the axes.
    # ax.plot(T, B, label='ude=0.5_erro',color = 'k')  # Plot some data on the axes.   
    # ax.plot(U, C, label='ude=1_erro',color = 'y')  # Plot some data on the axes.
    # ax.plot(Y, D, label='target_erro',color = 'b')  # Plot more data on the axes...




    plt.ylabel(j+' (m)')  #横坐标含义
    plt.xlabel('time(s)')  #纵坐标含义
    plt.title(j+' direction')
    plt.legend()
    plt.savefig(j+'.png',dpi=600,bbox_inches="tight")
    plt.show()
