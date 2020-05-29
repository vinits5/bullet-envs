import numpy as np 
import pybullet as p 
import pybullet_data

class Maze:

	def __init__(self):

		self.num_walls = 4
		self.obstacle_centers = [[2, 	 0],
								 [-2,	-2],
								 [-2,	 1],
								 [1.8, 	-2]]


	def generate_walls(self):
		wall_length = 5
		wall_width = 0.2

		center_x= np.array([5,  0, -5, 0])
		center_y= np.array([0, -5,  0, 5])

		half_height = 	np.array([1.0]*self.num_walls)
		half_length = 	np.array([wall_width, wall_length, wall_width, wall_length])
		half_width = 	np.array([wall_length, wall_width, wall_length, wall_width])

		for i in range(self.num_walls):

			box_id = p.createCollisionShape(
			p.GEOM_BOX,
			halfExtents=[half_length[i], half_width[i], half_height[i]])

			p.createMultiBody(
			baseMass=0,
			baseCollisionShapeIndex=box_id,
			basePosition=[center_x[i], center_y[i], half_height[i]])
			

	def generate_obstacles(self):
		for i in range(len(self.obstacle_centers)):
			p.loadURDF("./turtlebot_urdf/obstacles.urdf",basePosition=[self.obstacle_centers[i][0], self.obstacle_centers[i][1], 0.8])
