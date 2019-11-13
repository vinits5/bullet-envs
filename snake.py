import numpy as np
import pybullet as p
import pybullet_data
import time
import math

# Settings for the dynamics
FRICTION_VALUES = [1, 0.1, 0.01]
PI = math.pi
gravity = -9.8
timeStep = 1/100.0
_gaitSelection = 1
selfCollisionEnabled = True
motorVelocityLimit = np.inf
motorTorqueLimit = np.inf
kp = 10
kd = 0.1
initState = [0]*16
initPosition = [0]*3
initOrientation = [0,0,0,1]
SCALING_FACTOR = PI/2

# Render Settings
_cam_dist = 1.0
_cam_yaw = 0
_cam_pitch = -30
_cam_roll = 0
upAxisIndex = 2
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
fov = 60
nearVal = 0.1
farVal = 100

class Snake(object):
	#The snake class simulates a snake robot from HEBI
	def __init__(self, pybullet_client, urdf_root):

		self.numMotors = 16
		# self.link = 
		self._pybulletClient = pybullet_client

		self._urdf = urdf_root
		self._selfCollisionEnabled = selfCollisionEnabled
		self._motorVelocityLimit = motorVelocityLimit
		self._motorTorqueLimit = motorTorqueLimit
		self._kp = kp
		self._kd = kd
		self._timeStep = timeStep
		self._gaitSelection = _gaitSelection

		self._cam_dist = _cam_dist
		self._cam_yaw = _cam_yaw
		self._cam_pitch = _cam_pitch
		self._cam_roll = _cam_roll
		self.upAxisIndex = upAxisIndex
		self.fov = fov
		self.nearVal = nearVal
		self.farVal = farVal
		self.counter = 0
		# self._maxForce = np.inf

		# Create functions
		# if self._is_render:
		# 	self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
		# else:
		# 	self._pybullet_client = bullet_client.BulletClient()

	def buildMotorList(self):
		numJoints = self._pybulletClient.getNumJoints(self.snake)
		self.motorList = np.arange(3,(numJoints),3).tolist()
		# print("Moror List", self.motorList)

	def reset(self, hardReset):
		# Check for hard reset.
		if hardReset:
			self._pybulletClient.resetSimulation()
			self._pybulletClient.setAdditionalSearchPath(pybullet_data.getDataPath())
			self._pybulletClient.setGravity(0,0,gravity)
			self._pybulletClient.loadURDF("plane.urdf")
			self.snake = self._pybulletClient.loadURDF(self._urdf, [0, 0, 0], useFixedBase=0)
		else:
			self.buildMotorList()
			self.resetPositionOrientation()
			self.resetPose()
			# self.resetBaseVelocity()
		return True

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
			self._pybulletClient.resetJointState(self.snake, i, initState[count])
			count += 1
		# print(initState)

	def resetPositionOrientation(self):
		self._pybulletClient.resetBasePositionAndOrientation(self.snake, initPosition, initOrientation)

		pass
	def getBaseOrientation(self):
		_,orientation = self._pybulletClient.getBasePositionAndOrientation(self.snake)
		return orientation

	def getBasePosition(self):
		position,_ = self._pybulletClient.getBasePositionAndOrientation(self.snake)
		return position

	def getActionDimensions(self):
		return (len(self.motorList))

	def getObservationDimensions(self):
		return (len(self.motorList)*3 + 7)

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

	def getObservation(self):
		observation = np.array([0.0]*self.getObservationDimensions())
		observation[0:len(self.motorList)] = self.getPosition()
		observation[len(self.motorList):2*len(self.motorList)] = self.getVelocity()
		observation[2*len(self.motorList):3*len(self.motorList)] = self.getTorque()
		observation[3*len(self.motorList):3*len(self.motorList)+3] = self.getBasePosition()
		observation[3*len(self.motorList)+3:] = self.getBaseOrientation()
		return observation

	def applyActions(self, action):
		motorCommands = self.convertActionToJointCommand(action)
		self._pybulletClient.setJointMotorControlArray(self.snake, self.motorList, self._pybulletClient.POSITION_CONTROL, motorCommands)
		
	def convertActionToJointCommand(self, action):
		motorCommands = [i*SCALING_FACTOR for i in action]
		return motorCommands
		# pass

	def checkFeedback(self, action, observation):
		actionFeedback =[(action[i]*SCALING_FACTOR - observation[i])for i in range(len(self.motorList))]
		actionFeedback = np.asarray(actionFeedback)
		actionNorm = np.sqrt(actionFeedback.dot(actionFeedback))
		if actionNorm>0.02:
			return True
		else:
			return False

	def createAction(self,action):
		# numJoints = self._pybulletClient.getNumJoints(self.snake)
		action_ = [0]*16
		count = 0
		if _gaitSelection == 0:
			for i in range(0,self.numMotors,2):
				
				action_[i] = action[count]
				count +=1
				
		elif _gaitSelection == 1:
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
		action = self.createAction(action)
		self.counter = 0
		observation = self.getObservation()
		while self.checkFeedback(action,observation):
			self.applyActions(action)
			self._pybulletClient.stepSimulation()
			observation = self.getObservation()
			actionNorm = self.checkFeedback(action, observation)
			time.sleep(self._timeStep)
			self.counter += 1
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
																	   aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
																	   nearVal=self.nearVal,
																	   farVal=self.nearVal)
		(_, _, px, _, _) = self._pybulletClient.getCameraImage(width=RENDER_WIDTH,
												   height=RENDER_HEIGHT)
												   # viewMatrix=view_matrix,
												   # projectionMatrix=proj_matrix)
												   # renderer=self._pybulletClient.ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px)
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	
		