import numpy as np 
import pybullet as p 
import pybullet_data

class Maze:

	def __init__(self):

		self.num_walls = 4



	def generate_walls(self):

		center_x= np.array([5,0,-5,0])
		center_y= np.array([0,-5,0,5])
		half_height = np.array([3.5]*self.num_walls)
		half_length = np.array([0.2,5,0.2,5])
		half_width = np.array([5,0.2,5,0.2]*self.num_walls)

		for i in range(self.num_walls):

			box_id = p.createCollisionShape(
			p.GEOM_BOX,
			halfExtents=[half_length[i], half_width[i], half_height[i]])

			p.createMultiBody(
			baseMass=0,
			baseCollisionShapeIndex=box_id,
			basePosition=[center_x[i], center_y[i], half_height[i]])
			

	def generate_obstacles(self):

		b1 = p.loadURDF("../turtlebot_urdf/obstacles.urdf",basePosition=[2,0,0.8])
		b2 = p.loadURDF("../turtlebot_urdf/obstacles.urdf",basePosition=[-2,-2,0.8])
		b3 = p.loadURDF("../turtlebot_urdf/obstacles.urdf",basePosition=[-2,1,0.8])
		b4 = p.loadURDF("../turtlebot_urdf/obstacles.urdf",basePosition=[1.8,-2,0.8])
