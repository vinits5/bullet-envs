import pybullet as p
import time
import pybullet_data
import world_env as w_env
import numpy as np
import matplotlib.pyplot as plt


# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# offset = [0,0,0]
# p.setGravity(0,0,-10)
# hexaStartPos = [0,0,0.5]
# turtleStartOrientation = p.getQuaternionFromEuler([0,0,0])
# turtle = p.loadURDF("pybullet_robots/data/turtlebot.urdf",offset,turtleStartOrientation)
# # hexaId = p.loadURDF("urdf/m6.urdf",hexaStartPos,hexaStartOrientation)
# plane = p.loadURDF("plane.urdf")
# # block = p.loadURDF("single_block.urdf",[0,0,0.6])
# p.setRealTimeSimulation(1)

# w = w_env.Maze()
# w.generate_walls()
# w.generate_obstacles()

# # for j in range (p.getNumJoints(turtle)):
# # 	print(p.getJointInfo(turtle,j))
# forward=0
# turn=0

# ray_start = [0,0,1]
# ray_end = [0,0,-3]
# y_range = np.arange(-6,6,0.1)
# x_range = np.arange(0,8,0.1)


class LIDAR(object):

	def __init__(self):
		self.radius = 8
		self.ray_angle = np.arange(0,2*np.pi,0.1)

	def ray_generator(self,ray_start,ray_end):
		rayHitColor = [0,0,1]
		data_ray_cast = (p.rayTestBatch(rayFromPositions = ray_start, rayToPositions = ray_end))
		# p.addUserDebugLine(ray_start[0],ray_end[0], rayHitColor)
		# print(data_ray_cast)
		return data_ray_cast

	def plot3d(self,x,y,z):
		x_high = []
		y_high = []
		z_high = []

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		print(len(y))
		ax.scatter(x, y,z, c='b')
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')

		ax.set_xlim([-8,8]) 
		ax.set_ylim([-8,8])
		ax.set_zlim([0,0.5])
		plt.show()

	def set_ray(self,pos,quat):

		distance = []
		ray_start_vector = []
		ray_end_vector = []
		x,y,z = [],[],[]
		count = 0

		rotation_matrix = p.getMatrixFromQuaternion(quat)	
		rotation_matrix = np.array(rotation_matrix).reshape(3,3)
		rotation_matrix[2,:] = [0,0,1]
		ray_start = [pos[0],pos[1],pos[2]+0.45]
		
		for theta in self.ray_angle:
			ray_start_vector.append(ray_start)
			vector = np.array([self.radius*np.cos(theta),self.radius*np.sin(theta),0])
			vector = np.matmul(rotation_matrix,vector)
			ray_end= (np.array(ray_start)+vector).tolist()
			ray_end_vector.append(ray_end)
		data_ray_cast = self.ray_generator(ray_start_vector,ray_end_vector)

		for batch in range(len(data_ray_cast)):
			d = ((data_ray_cast[batch][3][0]-ray_start_vector[batch][0])**2 + (data_ray_cast[batch][3][1]-ray_start_vector[batch][1])**2 + (data_ray_cast[batch][3][2]-ray_start_vector[batch][2])**2)**0.5 
			distance.append(d)
			x.append(data_ray_cast[batch][3][0])
			y.append(data_ray_cast[batch][3][1])
			z.append(data_ray_cast[batch][3][2])

		return distance,x,y,z


if __name__ == "__main__":

	sensor = LIDAR()
	pos,quat = p.getBasePositionAndOrientation(turtle)
	_,x,y,z = sensor.set_ray(pos,quat)
	sensor.plot3d(x,y,z)

