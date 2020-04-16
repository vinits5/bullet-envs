import numpy as np
import pybullet_data
import time
import math
import os
import random
PI = math.pi

import world_env as w_env
import turtlebot_lidar as lidar

gravity = -9.8
timeStep = 1/100.0

class Turtlebot(object):
	#The Turtlebot class simulates a turtlebot robot by Willow Garage.
	def __init__(self, pybullet_client, urdf_root, args=None):
		self.numMotors = 2
		self._pybulletClient = pybullet_client
		self._urdf = urdf_root
		self._timeStep = timeStep
		self.counter = 0
		self.initPosition = [0,0,0]
		self.initOrientation = [0,0,0,1]
		
		self.MAX_TORQUE = 1000
		self.forces = [self.MAX_TORQUE]*self.numMotors
		self._speed = 10
		self.wall_boundary = [-4.8,4.8]
		self.obstacle_centers = [[2,0],[-2,-2],[-2,1],[1.8,-2]]
		self.w = w_env.Maze()
		self.lidar = lidar.LIDAR()
		self.goal_reached_threshold = 0.3
		self.threshold = self.goal_reached_threshold+0.2
		# if args is not None:
		# 	self.setParams(args)
		# else:
		# 	self.defaultParams()

	def setParams(self, args):
		self._motorVelocityLimit = args.motorVelocityLimit
		self._motorTorqueLimit = args.motorTorqueLimit

		self._cam_dist = args.cam_dist
		self._cam_yaw = args.cam_yaw
		self._cam_pitch = args.cam_pitch
		self._cam_roll = args.cam_roll
		self.upAxisIndex = args.upAxisIndex
		self.RENDER_HEIGHT = args.render_height
		self.RENDER_WIDTH = args.render_width
		self.fov = args.fov
		self.nearVal = args.nearVal
		self.farVal = args.farVal
		self.mode = args.mode

	def defaultParams(self):
		# Settings for the dynamics
		self._motorVelocityLimit = np.inf
		self._motorTorqueLimit = np.inf

		# Render Settings
		self._cam_dist = 5.0
		self._cam_yaw = 50
		self._cam_pitch = -35
		self._cam_roll = 0
		self.upAxisIndex = 2
		self.RENDER_HEIGHT = 720
		self.RENDER_WIDTH = 1280
		self.fov = 60
		self.nearVal = 0.1
		self.farVal = 100
		self.mode = 'train'

	def add_env(self):

		self.w.generate_walls()
		self.w.generate_obstacles()
		

	def buildMotorList(self):
		numJoints = 2					# Joints [0, 1] refers to wheel motors.
		self.motorList = np.arange(0, numJoints).tolist()

	def getGoalPosition(self):

		X = [random.uniform(self.wall_boundary[0],self.wall_boundary[1]),random.uniform(self.wall_boundary[0],self.wall_boundary[1])]

		for centers in self.obstacle_centers:
			if ((X[0]-(centers[0]))**2+ (X[1]-(centers[1]))**2) > (self.threshold)**2:
				self.goalPosition = X
				return self.goalPosition
			else:
				# print("generate points again")
				self.getGoalPosition()


	def reset(self, hardReset=False):
		# Check for hard reset.
		self.goalPosition = np.array([1, 1, 1])
		if hardReset:
			self._pybulletClient.resetSimulation()
			self._pybulletClient.setAdditionalSearchPath(pybullet_data.getDataPath())
			self._pybulletClient.setGravity(0, 0, gravity)
			self._pybulletClient.loadURDF("plane.urdf")
			self.turtlebot = self._pybulletClient.loadURDF(self._urdf, self.initPosition)
			self.add_env()
			# self.add_obstacle("block.urdf", [2, 0, 0.1])
		else:
			self.resetPositionOrientation()
		return True

	def resetPositionOrientation(self):
		self._pybulletClient.resetBasePositionAndOrientation(self.turtlebot, self.initPosition, self.initOrientation)

	def getBaseOrientation(self):
		_, orientation = self._pybulletClient.getBasePositionAndOrientation(self.turtlebot)
		return orientation

	def getBasePosition(self):
		position, _ = self._pybulletClient.getBasePositionAndOrientation(self.turtlebot)
		return position

	def getActionDimensions(self):
		return (len(self.motorList))

	def getObservationDimensions(self):
		return len(self.getObservation())

	def getObservationUpperBound(self):
		upperBound = np.array([1.0]*self.getObservationDimensions())
		upperBound *= self.wall_boundary[0]
		return upperBound

	def getObservationLowerBound(self):
		lowerBound = np.array([0.0]*self.getObservationDimensions())
		return lowerBound

	def getVelocity(self):
		velocity = np.array([0.0]*len(self.motorList))
		count = 0
		for i in self.motorList:
			_, velocity[count], _, _ = self._pybulletClient.getJointState(self.turtlebot, i)
			count += 1
		return velocity

	def getTorque(self):
		torque = np.array([0.0]*len(self.motorList))
		for idx, motorNo in enumerate(self.motorList):
			_,_,_,torque[idx]= self._pybulletClient.getJointState(self.turtlebot, motorNo)
		return torque

	def getForceInfo(self):
		torque = 0.0
		_,_,reactionForce,_= self._pybulletClient.getJointState(self.turtlebot, 0)  #0 is the head motor
		torque = reactionForce[2]
		return torque

	def getObservation(self):
		pos = self.getBasePosition()
		quat = self.getBaseOrientation()
		observation_position = [self.goalPosition[0]-pos[0],self.goalPosition[1]-pos[1]]
		observation_lidar,_,_,_ = self.lidar.set_ray(pos,quat)
		observation = observation_lidar+observation_position
		return observation
		

	def applyActions(self, action):
		leftWheelVelocity = action[0]*self._speed
		rightWheelVelocity = action[1]*self._speed
		self._pybulletClient.setJointMotorControl2(self.turtlebot, self.motorList[0], self._pybulletClient.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1000)
		self._pybulletClient.setJointMotorControl2(self.turtlebot, self.motorList[1], self._pybulletClient.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1000)

	def setTimeSteps(self):
		self._pybulletClient.setTimeStep(self._timeStep)

	def step(self, action):
		self.counter = 0
		self.applyActions(action)
		time.sleep(self._timeStep)
		self._pybulletClient.stepSimulation()
		observation = self.getObservation()
		return True
	
	def render(self):
		base_pos = self.getBasePosition()
		view_matrix = self._pybulletClient.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=base_pos,
			distance=self._cam_dist,
			yaw=self._cam_yaw,
			pitch=self._cam_pitch,
			roll=self._cam_roll,
			upAxisIndex=self.upAxisIndex)
		proj_matrix = self._pybulletClient.computeProjectionMatrixFOV(fov=self.fov,
																	   aspect=float(self.RENDER_WIDTH)/self.RENDER_HEIGHT,
																	   nearVal=self.nearVal,
																	   farVal=self.nearVal)

		cdist = 1.5
		cyaw = -30
		cpitch = -90
		p.resetDebugVisualizerCamera(cameraDistance=cdist, cameraYaw=cyaw, cameraPitch=cpitch, cameraTargetPosition=[1.28,0,0])
		(_, _, px, _, _) = self._pybulletClient.getCameraImage(width=self.RENDER_WIDTH,
												   height=self.RENDER_HEIGHT)
												   # viewMatrix=view_matrix,
												   # projectionMatrix=proj_matrix)
												   # renderer=self._pybulletClient.ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px)
		rgb_array = rgb_array[:, :, :3]
		# rgb_array[:,:,0], rgb_array[:,:,2] = rgb_array[:,:,2], rgb_array[:,:,0]
		return rgb_array