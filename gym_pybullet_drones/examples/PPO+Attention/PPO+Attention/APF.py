from typing import *
import numpy as np
from CWFobjects import *

class ApfVehicle(Vehicle):
    
    scale_repulse = 0.05
    r_perception = 3
    individual_balance = 3
    target_position = np.array([[2.1],[2.1]])
    origin_target_position = np.array([[10.1],[10.1]])
    min_dist = 0.1
    obs_influence_range = 10
    soft_influence_range = 2
    done = False
    orientation_last = np.array([[1],[0]])
    obstacle_closet = np.zeros((2,0))
    distance_to_target = 0

    def __init__(self, position, target_pos, scale) -> None:
        super().__init__(position, scale)
        self.target_position[0] = target_pos[0]
        self.target_position[1] = target_pos[1]
        self.target_position = self.target_position * scale
        self.origin_target_position = self.target_position
        self.origin_position = self.position

        self.scale_repulse = self.scale_repulse * scale **3
        self.r_perception = self.r_perception * scale
        self.min_dist = self.min_dist * scale
        self.obs_influence_range = self.obs_influence_range * scale
        self.individual_balance = self.individual_balance * scale

    def setTarget(self, target_pos):
        self.target_position[0] = target_pos[0]
        self.target_position[1] = target_pos[1]
    
    def distanceTarget(self):
        self.distance_to_target = np.linalg.norm(self.position - self.target_position)
        return self.distance_to_target

    def step(self, dt) -> None:
        if not self.done:
            super().step(dt)
            self.orientation_last = self.orientation
    
    def draw(self):
        tor = super().draw()
        plt.scatter(self.target_position[0, 0], self.target_position[1, 0], marker='*')
        return tor

    def attract(self, target_position):
        '''
        This function is used to calculate attractive force.
        Input:
            self_position: the position of pursuer
            target_position: the position of evader
        Output:
            F: the attractive force
        '''
        dist = np.linalg.norm(target_position - self.position)
        if dist < self.min_dist:
            dist = self.min_dist
        F = (target_position - self.position) / dist
        return F

    def findClosestObs(self, obstacles:List[Obstacle]):
        # find the nearest boundary point

        obstacle_total = np.zeros((2,0))
        for o in obstacles:
            obstacle_total = np.hstack((obstacle_total, o.getBoundary()))
        # the index of nearest obstacle
        temp = np.argmin(np.linalg.norm(obstacle_total - self.position, axis=0))
        # the position of the nearest obstacle
        obstacle_closest = obstacle_total[:, temp:temp+1]
        self.obstacle_closet = obstacle_closest
        return obstacle_closest

    def findClosestRep(self, obstacles:List[Obstacle]):
        # find the nearest boundary point
        obstacle_total = np.zeros((2,0))
        for o in obstacles:
            obstacle_total = np.hstack((obstacle_total, o.getBoundary()))
        # the index of nearest obstacle
        temp = np.argmin(np.linalg.norm(obstacle_total - self.position, axis=0))
        ob1 = obstacle_total[:, temp:temp+1]
        np.delete(obstacle_total, temp, axis=1)
        temp = np.argmin(np.linalg.norm(obstacle_total - self.position, axis=0))
        ob2 = obstacle_total[:, temp:temp+1]
        # the position of the nearest obstacle
        obstacle_closest = np.hstack((ob1, ob2))
        return obstacle_closest
    
    def findClosestObsWithFriend(self, obstacles:List[Obstacle], friends:List[Vehicle]):
        obstacle_total = np.zeros((2,0))
        for o in obstacles:
            obstacle_total = np.hstack((obstacle_total, o.getBoundary()))
        virtual_obstacle = np.zeros((2, 0))  # virtual obstacle buffer
        for i, v in enumerate(friends):
            virtual_obstacle = np.hstack((virtual_obstacle, v.position))
        # add virtual obstacles into obstacles list
        obstacle_with_other_agent = np.hstack((obstacle_total, virtual_obstacle))
        # the index of nearest obstacle (considering virtual obstacles)
        temp = np.argmin(np.linalg.norm(obstacle_with_other_agent - self.position, axis=0))
        # the position of nearest obstacle (considering virtual obstacles)
        obstacle_closest_with_other_agent = obstacle_with_other_agent[:, temp:temp+1]
        return obstacle_closest_with_other_agent

    def repulse(self, obstacles:List[Obstacle], friends:List[Vehicle]):
        '''
        This function is used to calculate repulsive force.
        Input:
            self_position: the position of pursuer
            obstacle_closest: the position of the nearest obstacle
            influence_range: the influence range of obstacles
            scale_repulse: the scale factor of repulsive force
        Output:
            F: the repulsive force
        '''

        '''
        virtual_obstacle = np.zeros((2, 0))  # virtual obstacle buffer
        for v in friends:
            virtual_obstacle = np.hstack((virtual_obstacle, v.position))

        obstacle_with_other_agent = np.hstack((self.obstacle_total, virtual_obstacle))
        '''
        obstacle_closest = self.findClosestRep(obstacles)[:, 0:1]
        # F = self.scale_repulse * (1 / (dist - 100) - 1 / obstacle.influence_range) / (dist - 100) ** 2 * (self.position - obstacle.position) / dist
        F = np.zeros((2,1))

        if np.linalg.norm(self.position - obstacle_closest) < self.obs_influence_range:
            dist = np.linalg.norm(self.position - obstacle_closest)
            if dist < self.min_dist:dist = self.min_dist
            # if the pursuer is within the obstacle's influence range
            for i in range(obstacle_closest.shape[1]):
                F += self.scale_repulse * (1 / (dist) - 1 / self.obs_influence_range) / (dist) ** 2 * (self.position - obstacle_closest[:,i:i+1]) / dist
            return F
        else:
            return np.array([[0], [0]])

    def soft_repulse(self, obstacles:List[Obstacle], friends:List[Vehicle]):
        obstacle_closest = self.findClosestRep(obstacles)[:, 0:1]
        return np.array([[0], [0]])
        
    def individual(self, friends:List[Vehicle]):
        '''
        This function is used to calculate inter-individual force.
        Input:
            self_position: the position of pursuer
            friend_position:  the positions of teammates
            individual_balance: lambda
            r_perception: d_s
        Output:
            F: the individual force
        '''
        F = np.zeros((2, 0))
        for friend in friends:
            dist = np.linalg.norm(friend.position - self.position)
            temp = (friend.position - self.position) / dist * (0.5 - self.individual_balance / dist)
            # if dist < self.r_perception:    # mask distant teammates
            F = np.hstack((F, temp)) 
        '''
        for i in range(friend_position.shape[1]):
            temp = (friend_position[:, i:i + 1] - self_position) / np.linalg.norm(
                friend_position[:, i:i + 1] - self_position) * (
                        0.5 - individual_balance / (np.linalg.norm(friend_position[:, i:i + 1] - self_position) - 200))
            if np.linalg.norm(friend_position[:, i:i + 1] - self_position) < r_perception:
                # mask distant teammates
                F = np.hstack((F, temp))
        '''
        if F.size == 0:
            F = np.zeros((2, 1))
        return np.mean(F, axis=1, keepdims=True)

    def wall_follow(self, F_repulse, F_individual):
        '''
        Wall following rules for pursuers.
        Input:
            self_orientation: the pursuer's heading
            F_repulse: the repulsive force of the pursuer
            F_individual: the inter-individual force of the pursuer
        Output:
            rotate_vector: the resultant force according to wall following rules
        '''
        # calculate n_1 and n_2
        rotate_matrix = np.array([[0, -1], [1, 0]])
        rotate_vector1 = np.matmul(rotate_matrix, F_repulse)
        rotate_vector2 = -1 * rotate_vector1
        # choose between n_1 and n_2
        temp1 = np.linalg.norm(rotate_vector1 - self.orientation)
        temp2 = np.linalg.norm(rotate_vector2 - self.orientation)
        if np.linalg.norm(F_individual) < 1:  # if inter-individual force is less threshold B
            if temp1 > temp2:  # choose according to the heading
                return rotate_vector2
            else:
                return rotate_vector1
        else:  # if inter-individual force exceeds threshold B,choose according to the inter-individual force
            if np.dot(np.ravel(rotate_vector1), np.ravel(F_individual)) > 0:  #
                return rotate_vector1
            else:
                return rotate_vector2

    def APF_decision(self, friends:List[Vehicle], obstacles:List[Obstacle]):
        '''
        This function is used to calculate the attractive force, the repulsive force, the inter-individual force.
        Input:
            self_position: the position of the pursuer
            friend_position: the positions of teammates
            target_position: the position of the evader
            obstacle_closest: the position of the nearest obstacle
            scale_repulse: the scale factor of repulsive force
            individual_balance: the parameter of inter-individual force, lambda
            r_perception: d_s
        Output:
            F_attract: the attractive forcce
            F_repulse: the repulsive force
            F_individual: the inter-individual force
            F: the resultant force of above three forces
        '''
        influence_range = 800  # the influence range of obstacles
        F_attract = self.attract(self.target_position)  # calculate the attractive force
        # calculate the repulsive force
        F_repulse = self.repulse(obstacles, friends)
        # calculate the inter-individual force
        F_individual = self.individual(friends)
        # calculate the resultant force
        F = F_attract + F_repulse + F_individual
        return F_attract, F_repulse, F_individual, F

    def total_decision(self, friends:List[Vehicle], obstacles:List[Obstacle]):
        '''
        This function is used to calculate resultant force, considering the wall following rules.
        Input:
            agent_position: the positions of pursuers
            agent_orientation: the headings of pursuers
            obstacle_closest: the positions of the closest obstacle for all pursuers
            target_position: the position of evader
            scale_repulse: the parameters(eta) for all pursuers
            individual_balance: the parameters (lambda) for all pursuers
            r_perception: d_s
        Output: the resultant force for all pursuers
        '''
        F = np.zeros((2, 0))  # resultant force buffer
        wall_following = np.zeros((1, 0))  # flag of whether pursuers move according to the wall following rules
        
        self_position = self.position
        self_orientation = self.orientation
        # calculate APF forces
        F_attract, F_repulse, F_individual, F_temp = self.APF_decision(friends, obstacles)
        vector1 = np.ravel(F_attract + F_repulse)  # calculate F_ar
        vector2 = np.ravel(F_attract)

        if np.any(F_repulse):
            if np.dot(vector1, vector2) < 0:  # if the angle between F_ar and F_a exceeds 90 degree
                # move according to wall following rules
                F_temp = self.wall_follow(F_repulse, F_individual)
                wall_following = True
            elif np.dot(np.ravel(F_attract), np.ravel(F_repulse)) < 0:
                # soft wall following
                if np.any(F_attract):
                    F_attract = F_attract / np.linalg.norm(F_attract)

                F_rotate = self.wall_follow(F_repulse, F_individual)
                if np.any(F_rotate):
                    F_rotate = F_rotate / np.linalg.norm(F_rotate)
                else:
                    F_rotate = np.zeros((2,1))
                F_temp = F_temp + 2*np.linalg.norm(F_repulse) * F_rotate
                # F_temp = F_attract + 1 * F_rotate
                wall_following = True
            else:
                wall_following = False
        
        F_norm = np.linalg.norm(F_temp)
        if F_norm != 0:
            F_temp = F_temp / np.linalg.norm(F_temp)  # normalize the resultant force

        F = F_temp
        return F, wall_following
