from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pts
import math

class CWFObject:
    position = np.array([[0],[0]])
    orientation = np.array([[1],[0]])
    velocity = np.array([[0],[0]])
    origin_position = np.zeros((2,1))
    origin_orientation = np.array([[1],[0]])
    scale = 1

    def setPosition(self, position):
        self.position = np.transpose(np.array(position).reshape(1,-1))
        self.origin_position = self.position

    def setVelocity(self, velocity):
        self.velocity = np.transpose(np.array(velocity).reshape(1,-1))

    def step(self, dt) -> None:
        self.position = self.position + self.velocity *dt

    @abstractmethod
    def draw(self):
        pass

class Obstacle(CWFObject):
    boundary = np.zeros((2,0))
    outer_boundary = np.zeros((2,0))
    preserve_dist = 0.1
    bound_step = 0.005

    def __init__(self, bound_step, scale=1) -> None:
        super().__init__()
        self.bound_step = bound_step * scale
        self.scale = scale
        self.preserve_dist = self.preserve_dist * scale

    def getBoundary(self):
        return self.boundary
    
    def getOuterBoundary(self):
        return self.outer_boundary

    @abstractmethod
    def refreshBoundary(self):
        pass
    
    def step(self, dt):
        super().step(dt)
        self.refreshBoundary()
    
    def draw(self):
        bound = self.getBoundary()
        # 实心填充绘制
        plt.fill(bound[0,:], bound[1,:], color="gray")
        # 边线绘制
        plt.plot(bound[0,:], bound[1,:],  color="black")

        # outer = self.getOuterBoundary()
        # 散点图形式绘制
        # plt.scatter(outer[0,:], outer[1,:], s=0.1)

class PolyObstacle(Obstacle):
    points = None

    def __init__(self, position, points, bound_step, scale=1) -> None:
        super().__init__(bound_step, scale)
        self.points = np.transpose(np.array(points))
        self.points = self.points * scale
        mean = np.mean(self.points, axis=1).reshape(1,-1).transpose()
        self.points -= mean * np.ones(self.points.shape[1])
        self.position = np.transpose(np.array(position).reshape(1,-1)) * scale + mean
        self.origin_position = self.position
        self.refreshBoundary()
    
    def refreshBoundary(self) -> None:
        # number of corner points
        num = self.points.shape[1]
        # boundary
        self.boundary = np.zeros((2,0))
        self.outer_boundary = np.zeros((2,0))
        for i in range(num - 1):
            point1 = self.points[:, i:i+1] + self.position
            point2 = self.points[:, i+1:i+2] + self.position
            nump = int(np.linalg.norm(point1 - point2) / self.bound_step)
            bound12 = np.hstack((np.linspace(point1[0], point2[0], nump), np.linspace(point1[1], point2[1], nump))).transpose()
            self.boundary = np.hstack((self.boundary, bound12))
        pointn = self.points[:, num-1:num] + self.position
        point1 = self.points[:, 0:1] + self.position
        nump = int(np.linalg.norm(point1 - point2) / self.bound_step)
        boundn1 = np.hstack((np.linspace(pointn[0], point1[0], nump), np.linspace(pointn[1], point1[1], nump))).transpose()
        self.boundary = np.hstack((self.boundary, boundn1))

        # Calculate Outer Boundary
        '''
        posb = self.position * np.ones((1, self.boundary.shape[1]))
        shap = self.boundary - posb
        absl = np.linalg.norm(shap, axis=0)
        mina = np.argmin(absl)
        prev = self.preserve_dist * np.ones_like(absl)
        absl = absl[mina] * np.ones_like(absl)
        shap = np.multiply(shap, np.divide((absl + prev),absl))
        self.outer_boundary = posb + shap
        '''

class CircleObstacle(Obstacle):
    radius = 0

    def __init__(self, position, radius, bound_step, scale=1) -> None:
        super().__init__(bound_step, scale)
        self.setPosition(position)
        self.position = self.position * scale
        self.radius = radius * scale
        self.origin_position = self.position
        self.refreshBoundary()

    def refreshBoundary(self):
        self.boundary = np.zeros((2,0))
        self.outer_boundary = np.zeros((2,0))
        num = int(2*math.pi*self.radius / self.bound_step)
        angles = np.linspace(0, 2*math.pi, num)
        for a in angles:
            point = np.array([self.radius*math.cos(a), self.radius*math.sin(a)]).reshape(-1,1)
            # outer_point = point * (self.radius + self.preserve_dist)/self.radius
            point = point + self.position
            # outer_point = outer_point + self.position
            self.boundary = np.hstack((self.boundary, point))
            # self.outer_boundary = np.hstack((self.outer_boundary, outer_point))

class QuadObstacle(PolyObstacle):

    def __init__(self, position, point1, point2, point3, point4, bound_step, scale) -> None:
        super().__init__(position, [point1, point2, point3, point4], bound_step, scale)
    
    def draw(self):
        bound = self.getBoundary()
        # 实心填充绘制
        # plt.fill(bound[0,:], bound[1,:], color="gray")
        # 边线绘制
        plt.plot(bound[0,:], bound[1,:],  color="black")

class Vehicle(CWFObject):
    type = 'none'
    trajectory = np.zeros((2,0))

    def __init__(self, position, scale=1) -> None:
        super().__init__()
        self.position = np.transpose(np.array(position).reshape(1,-1)) * scale
        self.scale = scale
    
    def draw(self):
        circle = pts.Circle(self.position, self.scale / 3.5)
        plt.quiver(self.position[0], self.position[1], self.orientation[0],
                               self.orientation[1], color='green', scale=10)
        return circle

    def step(self, dt) -> None:
        super().step(dt)
        self.trajectory = np.hstack((self.trajectory, self.position))