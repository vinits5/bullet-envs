import numpy as np
import pybullet as p


FRICTION_VALUES = [1, 0.1, 0.01]
PI = math.pi

class Snake(object):
	#The snake class simulates a snake robot from HEBI
	def __init__(self,
				 pybullet_client):

		self.numMotors = 16
		self.link = 
		self._pybulletClient = pybullet_client
		self._urdf = urdf_root
		self._selfCollisionEnabled = selfCollisionEnabled
		self._motorVelocityLimit = motorVelocityLimit
		self._motorTorqueLimit = motorTorqueLimit
		self._kp = kp
		self._kd = kd
		self._timeStep = timeStep
		self._gaitSelection = _gaitSelection
		self._maxForce = 

	def buildMotorList(self):
		numJoints = self._pybulletClient.getNumJoints(self.quadruped)
		numRevoluteJoints = int((numJoints-1)/3)
		if _gaitSelection == 0:
			self.motorList = np.arange(3,(numJoints+3),6).tolist()
		elif _gaitSelection == 1:
			self.motorList = np.arange(6,(numJoints+3),6).tolist()
		else:
			self.motorList = np.arange(3,(numJoints+3),3).tolist()

	def reset(self, reloadUrdf):
		if reloadUrdf:
			self.snake = PUT_URDF_ADDRESS_HERE
		self.buildMotorList()
		self.resetPose()
		self.resetPositionOrientation()
		self.resetBaseVelocity()


	def setDesiredMotorById(self, motorId, desiredAngle):
		self._pybullet_client.setJointMotorControl(bodyIndex=self.snake,
												jointIndex=motor_id,
												controlMode=self._pybullet_client.POSITION_CONTROL,
												targetPosition=desired_angle,
												positionGain=self._kp,
												velocityGain=self._kd,
												force=self._maxForce)

	def resetPose(self, self._initState):
		for i in range(len(self.motorList)):
			self._pybulletClient.resetJointState(self.snake, i, self._initState[i])


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
		upperBound = np.array([0.0]*self.getObservationDimensions)
		upperBound[0:self.motorList] = np.pi
		upperBound[self.motorList:2*self.motorList] = self._motorSpeedLimit
		upperBound[2*self.motorList:3*self.motorList] = self._motorTorqueLimit
		upperBound[3*self.motorList:] = 1.0;

		return upperBound

	def getObservationLowerBound(self):
		return -self.getObservationUpperBound


	def getPosition(self):
		position = np.array([0.0]*self.motorList)
		for i in range(self.motorList):
			position[i],_,_,_ = self._pybulletClient.getJointState(self.snake, i)
		return position

	def getvelocity(self):
		velocity = np.array([0.0]*self.motorList)
		for i in range(self.motorList):
			_,velocity[i],_,_,= self._pybulletClient.getJointState(self.snake, i)
		return velocity

	def getTorque(self):
		torque = np.array([0.0]*self.motorList)
		for i in range(self.motorList):
			_,_,torque[i],_,= self._pybulletClient.getJointState(self.snake, i)
		return torque

	def getObservation(self):
		observation = np.array([0.0]*self.getObservationDimensions)
		observation[0:self.motorList] = self.getPosition
		observation[self.motorList:2*self.motorList] = self.getvelocity
		observation[2*self.motorList:3*self.motorList] = self.getTorque
		observation[3*self.motorList:3*self.motorList+3] = self.getBasePosition
		observation[3*self.motorList+3:] = self.getBaseOrientation
		return observation

	def applyActions(self, action):
		self._pybulletClient.setJointMotorControlArray(self.snake, self.motorList, controlMode = POSITION_CONTROL, motor_commands)

	def convertActionToJointCommand(self, action):
		# return motor_commands
		pass

	def setTimeSteps(self, self._timeStep):
		self._pybulletClient.setTimeStep(self._timeStep)






	



