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

	def resetPose(self,self._initState):
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
		return (len(self.motorList)+7)

	def getObservationUpperBound(self):
		upperBound = np.array([0.0]*self.getObservationDimensions)
		upperBound[0:self.motorList] = np.pi
		upperBound[self.motorList:2*self.motorList] = self._motorSpeedLimit
		upperBound[2*self.motorList:3*self.motorList] = self._motorTorqueLimit
		upperBound[3*self.motorList:] = 1.0;

		return upperBound

	def getObservationLowerBound(self):
		return -self.getObservationUpperBound

	



