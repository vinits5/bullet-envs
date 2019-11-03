import numpy as np
import pybullet as p
import time
import math

FRICTION_VALUES = [1, 0.1, 0.01]
PI = math.pi
gravity = -9.8
timeStep = 1/240.0
_gaitSelection = 1
selfCollisionEnabled = True
motorVelocityLimit = np.inf
motorTorqueLimit = np.inf
kp = 10
kd = 0.1
initState = [0]*16
initPosition = [0]*3
initOrientation = [0,0,0,1]
class Snake(object):
	#The snake class simulates a snake robot from HEBI
	def __init__(self,
				 pybullet_client):

		self.numMotors = 16
		# self.link = 
		self._pybulletClient = pybullet_client
		
		# self._urdf = urdf_root
		self._selfCollisionEnabled = selfCollisionEnabled
		self._motorVelocityLimit = motorVelocityLimit
		self._motorTorqueLimit = motorTorqueLimit
		self._kp = kp
		self._kd = kd
		self._timeStep = timeStep
		self._gaitSelection = _gaitSelection
		# self._maxForce = np.inf

		# Create functions
		# if self._is_render:
		# 	self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
		# else:
		# 	self._pybullet_client = bullet_client.BulletClient()

	def buildMotorList(self):
		numJoints = self._pybulletClient.getNumJoints(self.snake)
		numRevoluteJoints = int((numJoints-1)/3)
		if _gaitSelection == 0:
			self.motorList = np.arange(3,(numJoints+3),6).tolist()
		elif _gaitSelection == 1:
			self.motorList = np.arange(6,(numJoints+3),6).tolist()
		else:
			self.motorList = np.arange(3,(numJoints+3),3).tolist()

	def reset(self, reloadUrdf):
		# Check for hard reset.
		if reloadUrdf:
			self.snake = PUT_URDF_ADDRESS_HERE
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
		count = 0
		for i in (self.motorList):
			_,_,_,torque[count]= self._pybulletClient.getJointState(self.snake, i)
			count += 1
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
		motorCommands = [i*PI/6 for i in action]
		return motorCommands
		# pass

	def setTimeSteps(self):
		self._pybulletClient.setTimeStep(self._timeStep)

	def step(self, action):
		# action = self.convertActionToJointCommand(action)
		self.applyActions(action)
		self._pybulletClient.stepSimulation()
		time.sleep(self._timeStep)
		return True
		pass