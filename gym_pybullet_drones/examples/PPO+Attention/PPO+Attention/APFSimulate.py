from typing import *
from matplotlib import pyplot as plt
from APF import ApfVehicle
from CWFobjects import *
import time,  random, copy

class APFSimulator:
    scale = 600

    minX = -5*scale
    maxX = 5*scale
    minY = 0*scale
    maxY = 10*scale
    num = 40

    v_show = ApfVehicle([0, 0], [0, 10], scale)
    '''
    o1 = QuadObstacle([1,1], [-0.5,-0.1], [-0.5, 0.1], [0.5,0.1],[0.5,-0.1], 0.01, scale)
    o2 = PolyObstacle([2.0,2.0], [[0, 0], [0.5, 0], [0.2, 0.3]], 0.01, scale)
    o3 = CircleObstacle([1.5, 1.5], 0.2, 0.01, scale)
    o4 = CircleObstacle([0.5, 1.5], 0.2, 0.01, scale)
    o5 = CircleObstacle([1.5, 0.5], 0.2, 0.01, scale)
    o6 = CircleObstacle([0.8, 2.2], 0.2, 0.01, scale)
    o7 = CircleObstacle([1.5, 2.2], 0.2, 0.01, scale)
    '''
    
    o1 = CircleObstacle([-2, 7], 0.5, 0.05, scale)
    o2 = CircleObstacle([0, 7], 0.5, 0.05, scale)
    o3 = CircleObstacle([2, 7], 0.5, 0.05, scale)
    o4 = CircleObstacle([-1, 5], 0.5, 0.05, scale)
    o5 = CircleObstacle([1, 5], 0.5, 0.05, scale)
    o6 = CircleObstacle([-2, 3], 0.5, 0.05, scale)
    o7 = CircleObstacle([0, 3], 0.5, 0.05, scale)
    o8 = CircleObstacle([2, 3], 0.5, 0.05, scale)
    oo = QuadObstacle([0., 5.], [-3.5, -5.], [-3.5, 5.], [3.5, 5.], [3.5, -5.], 0.05, scale)

    v1 = ApfVehicle([0, 1], [0, 8], scale)
    # v2 = ApfVehicle([0, 10], [0, 0], scale)
    v2 = ApfVehicle([0, 9], [0, 2], scale)
    v3 = ApfVehicle([-3, 5], [2, 5], scale)
    v4 = ApfVehicle([3, 5], [-2, 5], scale)

    obstacles : List[Obstacle] = [o1, o2, o3, o4, o5, o6, o7, o8, oo]
    vehicles:List[ApfVehicle] = [v1, v2]

    num_state = 4 + 1 + (len(vehicles) - 1) * 2  # dimension of state space
    wall_following = np.zeros((1, len(vehicles)))  # whether pursuers move according to wall following rules
    mode = 0
    num_agent = len(vehicles)
    num_action = 24
    v = 300
    delta_t = 0.1
    gamma = 0
    r_perception = 2000

    t = 0

    def __init__(self, gamma, mode):
        '''
        The initialization function.
        Input:
            gamma: the discount factor
            mode: whether the environment is used to train policies or evaluate policies. 'Train' means training mode,
                  the pursuer who collides will be move to the original position. 'Valid' means validation mode, the
                  pursuer who collides will quit the game.
        '''
        self.mode = mode  # training mode or validation mode
        self.num_agent = len(self.vehicles)  # number of pursuers
        self.num_action = 24  # number of discretized actions
        self.num_state = 4 + 1 + (self.num_agent - 1) * 2  # dimension of state space
        self.t = 0  # timestep
        self.v = 300  # velocity of pursuers (mm/s)
        self.delta_t = 0.1  # time interval (s)
        
        self.gamma = gamma  # discount factor
        self.r_perception = 2000  # sense range of pursuers

    def run(self, rate, num):
        t0 = time.time()
        i = 0
        notdone = True
        numv = len(self.vehicles)
        while( i < num and notdone):
            i = i + 1
            t1 = time.time()
            dt = t1 - t0
            for o in self.obstacles:
                # o.setVelocity([(random.random()-0.5)/10*self.scale, (random.random()-0.5)/10*self.scale])
                o.step(dt)
            for index, v in enumerate(self.vehicles):
                friends = self.vehicles[:index] + self.vehicles[index+1:numv]
                F, wf = v.total_decision(friends, self.obstacles)
                v.velocity = F * self.scale / 1
                v.orientation = F
            donenum = 0
            for v in self.vehicles:
                v.step(dt)
                if v.distanceTarget() < 0.1 * self.scale:
                    donenum += 1
            if donenum >= len(self.vehicles):
                notdone = False
            self.render(False)
            t0 = t1
        self.render(True)
        plt.show()                

    def render(self, drawField:bool=False, reward=0, action=0):
        plt.figure(1, figsize=(9,9))
        plt.cla()
        ax = plt.gca()
        plt.ylim(self.minY, self.maxY)
        plt.xlim(self.minX, self.maxX)
        ax.set_aspect(1)

        for o in self.obstacles:
            o.draw()

        for v in self.vehicles:
            ax.add_patch(v.draw())
            plt.plot(v.trajectory[0,:], v.trajectory[1,:])
        
        if drawField:
            self.v_show = copy.copy(self.v1)
            Xarea = np.linspace(self.minX, self.maxX, self.num)
            Yarea = np.linspace(self.minY, self.maxY, self.num)
            Xa, Ya = np.meshgrid(Xarea, Yarea)
            qpos = np.zeros((2,0))
            qdir = np.zeros((2,0))
            for (x,y) in zip(Xa.flatten(),Ya.flatten()):
                self.v_show.setPosition([x, y])
                qpos = np.hstack((qpos, self.v_show.position))
                # F_attract, F_repulse, F_individual, F = self.v_show.APF_decision([], self.obstacles)
                F, wall_following = self.v_show.total_decision([], self.obstacles)
                qdir = np.hstack((qdir, F))
            xq = qpos[0,:]
            yq = qpos[1,:]
            uq = qdir[0,:]
            vq = qdir[1,:]
            plt.quiver(xq, yq, uq, vq, scale=self.num)
        plt.title("reward:"+str(reward) + '\n' + "action:"+str(action))
        plt.show(block=False)
        # plt.savefig(str(self.t))#whether save figures
        plt.pause(0.001)
    
    def reward(self):
        numv = len(self.vehicles)
        reward = np.zeros((1, numv))  # reward buffer
        done = np.zeros((1, numv))  # done buffer
        position_buffer = np.zeros((2, numv))
        for index, v in enumerate(self.vehicles):
            position_buffer[:, index:index+1] = v.position
            done[0, index] = v.done

        for index, v in enumerate(self.vehicles):
            v.findClosestObs(self.obstacles)
            reward2 = 0  # r_time
            reward3 = 0  # r_tm
            reward4 = 0  # r_o
            reward5 = 0  # r_pot
            if v.done:
                success_range = 300  # d_c
                done_temp = 1
                reward1 = 0
            else:
                success_range = 200
            if np.linalg.norm(v.position - v.target_position) < success_range:  # if the distance the evader is less than d_c
                '''
                alldist = np.linalg.norm(v.origin_position - v.target_position)
                reward1 = 200 - v.trajectory.shape[1] / alldist * 1000  # r_main
                if reward1 < 100:
                    reward1 = 100
                '''
                reward1 = 100
                done_temp = 1.  # the pursuer captures the evader successfully
                v.done = True
                # if self.mode == 'Train':
                    # the pursuer collides and be moved to its original position
                    # position_buffer[:, index:index + 1] = v.origin_position
                    # v.trajectory = np.zeros((2,0))
            else:
                reward1 = 0
                done_temp = 0.
                normor = np.linalg.norm(v.orientation)
                if normor == 0:
                    normor = 1
                normor_last = np.linalg.norm(v.orientation_last)
                if normor_last == 0:
                    normor_last = 1
                if np.arccos(np.clip(np.dot(np.ravel(v.orientation_last), np.ravel(v.orientation)) / normor_last / normor,
                                     -1, 1)) > np.radians(45):  # if the pursuer's steering angle exceeds 45
                    # reward2 = -5
                    reward2 = -1
                else:
                    reward2 = 0

                if np.linalg.norm(v.position - v.obstacle_closet) > 200:
                    # if the distance from the nearest obstacle exceeds 200 mm
                    reward3 = 0
                elif np.linalg.norm(v.position - v.obstacle_closet) < 150:
                    # if the distance from the nearest obstacle is less than 150 mm
                    reward3 = -20
                    if self.mode == 'Train':
                        # the pursuer collides and be moved to its original position
                        position_buffer[:, index:index + 1] = v.origin_position
                        v.trajectory = np.zeros((2,0))
                        for o in self.obstacles:
                            o.position = o.origin_position
                        done_temp = 3.
                    if self.mode == 'Valid':
                        # the pursuer is inactive
                        done_temp = 3.
                else:
                    reward3 = -2
                
                dist = -1
                if numv > 1:
                    # find nearest teammate
                    friends = self.vehicles[:index] + self.vehicles[index+1:numv]
                    for friend in friends:
                        dist1 = np.linalg.norm(v.position - friend.position)
                        if dist < 0 or dist1 < dist:
                            dist = dist1
                    
                    # if the distance from the nearest teammate exceeds 200 mm
                    if dist > 200:
                        reward4 = 0
                    else:
                        reward4 = -10
                        if self.mode == 'Train':
                            # the pursuer collides and be moved to its original position
                            position_buffer[:, index:index + 1] = v.origin_position
                            v.trajectory = np.zeros((2,0))
                            done_temp = 3.
                        if self.mode == 'Valid':
                            # the pursuer is inactive
                            done_temp = 3.
                else:
                    reward4 = 0
                
                dist_t = v.distanceTarget()
                
                if dist_t < 5000:
                    potential = (5000 - dist_t) / 50000
                    # potential = 0
                else:
                    potential = 0
                
                reward5 = potential

                # if self.t == 1000:
                    # done_temp = 2.  # pursuers failed though no collision

            reward[0, index] = reward1 + reward2 + reward3 + reward4 + reward5  # the total reward
            done[0, index] = done_temp
        
        # in the training mode, initialize the collided pursuer, in validation mode, do nothing
        for index, v in enumerate(self.vehicles):
            v.position = position_buffer[:, index:index+1]
            v.done = done[0, index]

        return reward, done
    
    def step(self, action):
        '''
        Execute actions and receive reward signals from the environment.
        Input:
            action: the action indexes for all pursuers
        Output:
            self.state: the observations for all pursuers
            reward: the reward signals for all pursuers
            done: whether pursuers are inactive. For each pursuer, 0 means it is active, 1 means it has captured the
                  evader, 2 means the episode reaches the maximal length, 3 means it collides.

        '''
        self.t += 1
        ######agent#########
        numv = len(self.vehicles)
        scale_repulse = np.zeros((1, numv))  # eta buffer
        individual_balance = np.zeros((1, numv))  # lambda buffer
        Field = np.zeros((2,0))
        agent_position_buffer = np.zeros((2, numv))
        for i, v in enumerate(self.vehicles):  # transform the action indexes into parameter pairs
            if action[0, i] < 8:
                v.scale_repulse = 0
            elif action[0, i] < 16:
                v.scale_repulse = 4e8
            else:
                v.scale_repulse = 4e9
            if action[0, i] % 8 == 0:
                v.individual_balance = 30
            elif action[0, i] % 8 == 1:
                v.individual_balance = 100
            elif action[0, i] % 8 == 2:
                v.individual_balance = 250
            elif action[0, i] % 8 == 3:
                v.individual_balance = 500
            elif action[0, i] % 8 == 4:
                v.individual_balance = 750
            elif action[0, i] % 8 == 5:
                v.individual_balance = 1000
            elif action[0, i] % 8 == 6:
                v.individual_balance = 2000
            elif action[0, i] % 8 == 7:
                v.individual_balance = 3000
            v.r_perception = self.r_perception
            friends = self.vehicles[:i] + self.vehicles[i+1:numv]
            Fi, walli = v.total_decision(friends, self.obstacles)
            # calculate APF resultant forces
            Field = np.hstack((Field, Fi))
            if v.done:
                agent_position_buffer[:, i:i + 1] = np.array([[0], [0]])
            else:
                v.velocity = Fi * self.v
                agent_position_buffer[:, i:i + 1] = v.velocity * self.delta_t

        #####update#####
        for i, v in enumerate(self.vehicles):
            v.position = v.position + agent_position_buffer[:, i:i + 1]     # update pursuers'positions
            v.orientation_last = v.orientation
            v.orientation = Field[:, i:i+1]                 # update pursuers' headings
            v.trajectory = np.hstack((v.trajectory, v.position))

        for o in self.obstacles:
            # o.setVelocity([(random.random()-0.5)/5*self.scale, (random.random()-0.5)/5*self.scale])
            o.step(self.delta_t)

        reward, done = self.reward()  # calculate reward function
        self.update_state(done)  # update environment's state
        return self.state, reward, done

    def step_ppo(self, action):
        '''
        Execute actions and receive reward signals from the environment.
        Input:
            action: the action indexes for all pursuers
        Output:
            self.state: the observations for all pursuers
            reward: the reward signals for all pursuers
            done: whether pursuers are inactive. For each pursuer, 0 means it is active, 1 means it has captured the
                  evader, 2 means the episode reaches the maximal length, 3 means it collides.

        '''
        self.t += 1
        ######agent#########
        numv = len(self.vehicles)
        scale_repulse = np.zeros((1, numv))  # eta buffer
        individual_balance = np.zeros((1, numv))  # lambda buffer
        Field = np.zeros((2,0))
        agent_position_buffer = np.zeros((2, numv))
        # action = (np.tanh(action) + 1) * 10
        for i, v in enumerate(self.vehicles):  # transform the action indexes into parameter pairs
            # v.scale_repulse = math.exp((action[0, i:i+1] + 0))
            v.scale_repulse = (action[0, i:i+1] + 0) * 1e6
            v.individual_balance = (action[1, i:i+1]) * 40

            v.r_perception = self.r_perception
            friends = self.vehicles[:i] + self.vehicles[i+1:numv]
            Fi, walli = v.total_decision(friends, self.obstacles)
            # calculate APF resultant forces
            Field = np.hstack((Field, Fi))
            if v.done:
                agent_position_buffer[:, i:i + 1] = np.array([[0], [0]])
            else:
                v.velocity = Fi * self.v
                agent_position_buffer[:, i:i + 1] = v.velocity * self.delta_t

        #####update#####
        for i, v in enumerate(self.vehicles):
            v.position = v.position + agent_position_buffer[:, i:i + 1]     # update pursuers'positions
            v.orientation_last = v.orientation
            v.orientation = Field[:, i:i+1]                 # update pursuers' headings
            v.trajectory = np.hstack((v.trajectory, v.position))

        for o in self.obstacles:
            # o.setVelocity([(random.random()-0.5)/5*self.scale, (random.random()-0.5)/5*self.scale])
            o.step(self.delta_t)

        reward, done = self.reward()  # calculate reward function

        # punish when action imnormal
        # '''
        for i in range(reward.shape[1]):
            if action[0, i] < 0.01 or action[0, i] > 50:
                reward[0, i] -= 10
            if action[1, i] < 0.01 or action[0, i] > 50:
                reward[0, i] -= 10
        # '''

        self.update_state(done)  # update environment's state
        return self.state, reward, done
    
    def update_state(self, done):
        '''
        Update the environment state (self.state and other class properties).
        '''
        numv = len(self.vehicles)
        self.state = np.zeros((self.num_state, numv))  # clear the environment state

        for i, v in enumerate(self.vehicles):
            friends = self.vehicles[:i] + self.vehicles[i+1:numv]
            v_to_obs = v.findClosestObsWithFriend(self.obstacles, friends) - v.position
            normor = np.linalg.norm(v.orientation)
            if normor == 0:
                normor = 1
            # the bearing of the nearest obstacle
            angle_to_obs = np.arccos(
                np.clip(
                    np.dot(np.ravel(v_to_obs), np.ravel(v.orientation)) / np.linalg.norm(
                        v_to_obs) / normor, -1, 1)) / np.pi
            if np.cross(np.ravel(v.orientation), np.ravel(v_to_obs)) > 0:
                pass
            else:
                angle_to_obs = -angle_to_obs
            
            v_to_target = v.target_position - v.position
             # the bearing of target
            angle_to_target = np.arccos(
                np.clip(
                    np.dot(np.ravel(v_to_target), np.ravel(v.orientation)) / np.linalg.norm(
                        v_to_target) / normor, -1, 1)) / np.pi
            if np.cross(np.ravel(v.orientation), np.ravel(v_to_target)) > 0:
                pass
            else:
                angle_to_target = -angle_to_target

            state = np.zeros((self.num_state,))  # state buffer
            state[:4] = np.array([np.linalg.norm(v_to_obs) / 5000, angle_to_obs, np.linalg.norm(v_to_target) / 5000, angle_to_target],
                                 dtype='float32')  # update state

            for j, f in enumerate(friends):
                v_to_friend = f.position - v.position
                distance = np.linalg.norm(v_to_friend)
                # the bearing of teammate
                angle = np.arccos(
                    np.clip(
                        np.dot(np.ravel(v_to_friend), np.ravel(v.orientation)) / distance / normor, -1, 1)) / np.pi
                if np.cross(np.ravel(v.orientation), np.ravel(v_to_friend)) > 0:
                    pass
                else:
                    angle = -angle
                # mask distant teammates and update state
                state[5 + 2 * j] = distance / 5000
                state[6 + 2 * j] = np.array(angle)

            
            if np.any(done == 1):
                state[4] = 1
            else:
                state[4] = 0
            self.state[:, i] = state

    def reset(self):
        '''
        Reset the environment.
        Output:
            state: the observations for all pursuers
        '''
        self.t = 0
        for o in self.obstacles:
            o.position = o.origin_position
            o.refreshBoundary()
            o.orientation = np.zeros((2,1))
        
        for v in self.vehicles:
            # initialize pursuers' poisitions and headings
            # '''
            if self.mode == "Train":
                correct_position = False
                while not correct_position:
                    tempposition = np.random.random((2, 1))
                    tempposition[0, 0] = (tempposition[0, 0] - 0.5) * v.scale * 6.5
                    tempposition[1, 0] = (tempposition[1, 0] + 0.05) * v.scale * 9
                    correct_position = True
                    for o in self.obstacles:
                        if np.linalg.norm(tempposition - o.position) < o.scale:
                            correct_position = False
                    for v1 in self.vehicles:
                        if np.linalg.norm(tempposition - v1.position) < v.scale:
                            correct_position = False
            
                    temptarget = np.random.random((2, 1))
                    temptarget[0, 0] = (temptarget[0, 0] - 0.5) * v.scale * 6.5
                    temptarget[1, 0] = (temptarget[1, 0] + 0.05) * v.scale * 9
                    for o in self.obstacles:
                        if np.linalg.norm(temptarget - o.position) < o.scale:
                            correct_position = False
                    for v1 in self.vehicles:
                        if np.linalg.norm(temptarget - v1.target_position) < v.scale:
                            correct_position = False
                        if np.linalg.norm(temptarget - v1.position) < v.scale * 2:
                            correct_position = False
                    if np.linalg.norm(temptarget - tempposition) < v.scale * 6:
                        correct_position = False
                v.position = tempposition
                v.target_position = temptarget
                v.origin_position = v.position
            elif self.mode == "Valid":
                v.position = v.origin_position
                v.target_position = v.origin_target_position
            v.orientation = np.array([[1],[0]])
            v.orientation = v.origin_orientation
            v.orientation_last = v.orientation
            v.trajectory = np.zeros((2,0))
        
        for o in self.obstacles:
            o.position = o.origin_position
            o.orientation = np.zeros((2,1))

        self.done = np.zeros((1, len(self.vehicles)))  # whether pursuers is inactive

        self.update_state(self.done)  # update environment's state

        return self.state


if __name__ == "__main__":
    sim = APFSimulator(0.99, 'Valid')
    sim.reset()
    sim.run(1, 1000)