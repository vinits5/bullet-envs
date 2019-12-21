import numpy as np
import pybullet as p
import pybullet_data
import time
import math
PI = math.pi
gravity = -9.8
timeStep = 1/100.0


class Snake(object):
	#The snake class simulates a snake robot from HEBI
	def __init__(self, pybullet_client, urdf_root, args=None):

		self.numMotors = 16
		self._pybulletClient = pybullet_client
		self._urdf = urdf_root
		self._timeStep = timeStep
		self.counter = 0
		self.START_POSITION = [0,0,0]
		self.initState = [0]*16
		self.initPosition = [0]*3
		self.initOrientation = [0,0,0,1]
		self.FRICTION_VALUES = [1, 0.1, 0.01]
		self.MAX_TORQUE = 3 
		self.forces = [self.MAX_TORQUE]*self.numMotors

		if args is not None:
			self.setParams(args)
		else:
			self.defaultParams()

	def setParams(self, args):
		self._selfCollisionEnabled = args.selfCollisionEnabled
		self._motorVelocityLimit = args.motorVelocityLimit
		self._motorTorqueLimit = args.motorTorqueLimit
		self._kp = args.kp
		self._kd = args.kd
		self._gaitSelection = args.gaitSelection
		self.SCALING_FACTOR = PI/(args.scaling_factor*1.0)

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
		self._selfCollisionEnabled = True
		self._motorVelocityLimit = np.inf
		self._motorTorqueLimit = np.inf
		self._kp = 10
		self._kd = 0.1
		self._gaitSelection = 1
		self.SCALING_FACTOR = PI/6

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

	def buildMotorList(self):
		numJoints = self._pybulletClient.getNumJoints(self.snake)
		self.motorList = np.arange(3,(numJoints),3).tolist()
		# print("Moror List", self.motorList)

	def add_obstacle(self, position):
		self.obstacle = self._pybulletClient.loadURDF("../snake/block.urdf", basePosition=position, useFixedBase=0)

	def reset(self, hardReset):
		# Check for hard reset.
		if hardReset:
			self._pybulletClient.resetSimulation()
			self._pybulletClient.setAdditionalSearchPath(pybullet_data.getDataPath())
			self._pybulletClient.setGravity(0,0,gravity)
			self._pybulletClient.loadURDF("plane.urdf")
			self.snake = self._pybulletClient.loadURDF(self._urdf, [0, 0, 0], useFixedBase=0, flags=self._pybulletClient.URDF_USE_SELF_COLLISION)
			self.add_obstacle([2, 0, 0.1])
			self.setDynamics()
		else:
			self.buildMotorList()
			self.resetPositionOrientation()
			self.resetPose()
			# self.resetBaseVelocity()
		return True

	def setDynamics(self):
		self._pybulletClient.changeDynamics(self.snake, -1, lateralFriction=2, anisotropicFriction=self.FRICTION_VALUES)
		for i in range(self._pybulletClient.getNumJoints(self.snake)):
			self._pybulletClient.changeDynamics(self.snake, i, lateralFriction=2, anisotropicFriction=self.FRICTION_VALUES)
			self._pybulletClient.enableJointForceTorqueSensor(self.snake,i,1)

	def setDesiredMotorById(self, motorId, desiredAngle):
		self._pybulletClient.setJointMotorControl2(bodyIndex=self.snake,
												jointIndex=motorId,
												controlMode=self._pybulletClient.POSITION_CONTROL,
												targetPosition=desiredAngle,
												positionGain=self._kp,
												velocityGain=self._kd,
												)
		self._pybulletClient.stepSimulation()

	def resetPose(self):
		count = 0
		for i in (self.motorList):
			self._pybulletClient.resetJointState(self.snake, i, self.initState[count])
			count += 1
		# print(self.initState)

	def resetPositionOrientation(self):
		self._pybulletClient.resetBasePositionAndOrientation(self.snake, self.initPosition, self.initOrientation)

		pass
	def getBaseOrientation(self):
		_,orientation = self._pybulletClient.getBasePositionAndOrientation(self.snake)
		return orientation

	def getBasePosition(self):
		position,_ = self._pybulletClient.getBasePositionAndOrientation(self.snake)
		return position

	def getActionDimensions(self):
		return (len(self.motorList))


	#changing from 55 to 56
	def getObservationDimensions(self):
		return (len(self.motorList)*3 + 8)

	def getObservationUpperBound(self):
		upperBound = np.array([0.0]*self.getObservationDimensions())
		
		upperBound[0:len(self.motorList)] = np.pi
		upperBound[len(self.motorList):2*len(self.motorList)] = self._motorVelocityLimit
		upperBound[2*len(self.motorList):3*len(self.motorList)] = self._motorTorqueLimit
		upperBound[3*len(self.motorList):] = 1.0;

		return upperBound

	def getObservationLowerBound(self):
		return -self.getObservationUpperBound()


	def getPosition(self):
		position = np.array([0.0]*len(self.motorList))
		count = 0
		for i in (self.motorList):
			position[count],_,_,_ = self._pybulletClient.getJointState(self.snake, i)
			count += 1
		return position

	def getVelocity(self):
		velocity = np.array([0.0]*len(self.motorList))
		count = 0
		for i in (self.motorList):
			_,velocity[count],_,_,= self._pybulletClient.getJointState(self.snake, i)
			count += 1
		return velocity

	def getTorque(self):
		torque = np.array([0.0]*len(self.motorList))
		for idx, motorNo in enumerate(self.motorList):
			_,_,_,torque[idx]= self._pybulletClient.getJointState(self.snake, motorNo)
		return torque

	def getForceInfo(self):
		torque = 0.0
		_,_,reactionForce,_= self._pybulletClient.getJointState(self.snake, 0)  #0 is the head motor
		torque = reactionForce[2]
		return torque


	def getObservation(self):
		observation = np.array([0.0]*self.getObservationDimensions())
		observation[0:len(self.motorList)] = self.getPosition()
		observation[len(self.motorList):2*len(self.motorList)] = self.getVelocity()
		observation[2*len(self.motorList):3*len(self.motorList)] = self.getTorque()
		observation[3*len(self.motorList):3*len(self.motorList)+3] = self.getBasePosition()
		observation[3*len(self.motorList)+3:3*len(self.motorList)+7] = self.getBaseOrientation()
		observation[55] = self.getForceInfo()
		return observation

	def applyActions(self, action):
		motorCommands = self.convertActionToJointCommand(action)
		self._pybulletClient.setJointMotorControlArray(self.snake, self.motorList, self._pybulletClient.POSITION_CONTROL, motorCommands, forces=self.forces)
		
	def convertActionToJointCommand(self, action):
		motorCommands = [i*self.SCALING_FACTOR for i in action]
		return motorCommands
		# pass

	def checkFeedback(self, action, observation):
		actionFeedback =[(action[i]*self.SCALING_FACTOR - observation[i])for i in range(len(self.motorList))]
		actionFeedback = np.asarray(actionFeedback)
		actionNorm = np.sqrt(actionFeedback.dot(actionFeedback))
		if actionNorm>0.05:
			return True
		else:
			return False

	def createAction(self, action):
		# numJoints = self._pybulletClient.getNumJoints(self.snake)
		action_ = [0]*16
		count = 0
		if self._gaitSelection == 0:
			for i in range(0,self.numMotors,2):
				
				action_[i] = action[count]
				count +=1
				
		elif self._gaitSelection == 1:
			for i in range(1,self.numMotors,2):
				
				action_[i] = action[count]
				count +=1
		else:
			for i in range(0,self.numMotors,1):
				
				action_[i] = action[count]
				count +=1
		
		# print("Motor List", action_)
		return action_

	def setTimeSteps(self):
		self._pybulletClient.setTimeStep(self._timeStep)

	def step(self, action):
		if self.mode == 'test':
			self.imgs = []
			self.step_internal_observations = []
			
		action = self.createAction(action)
		self.counter = 0
		observation = self.getObservation()
		while self.checkFeedback(action,observation):
			self.applyActions(action)
			self._pybulletClient.stepSimulation()
			#if self.mode == 'test':
			#	if self.counter%4 == 0: self.imgs.append(self.render())
			observation = self.getObservation()
			if self.mode == 'test': self.step_internal_observations.append(observation)
			actionNorm = self.checkFeedback(action, observation)
			time.sleep(self._timeStep)
			self.counter += 1
			if self.counter > 40:	# To avoid extra steps and collision.
				break
		# print(self.counter)
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

	def calculateEnergy(self,observation):
		energy = []
		for i in range(self.numMotors):
			energy.append(observation[self.numMotors+i]*observation[2*(self.numMotors)+i]*self._timeStep)
		energy_total = np.sum(energy)
		return energy_total