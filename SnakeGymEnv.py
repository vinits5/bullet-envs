import gym

class SnakeGymEnv(gym.Env):
	def __init__(self, robot, NUM_MOTORS):
		self.robot = robot
		self.NUM_MOTORS = NUM_MOTORS
		self._action_bound = 1
		self.OBSERVATION_TORQUE_LIMIT = np.inf
		self.MOTOR_SPEED_LIMIT = np.inf

	def reset(self):
		observation = []
		return observation

	def step(self, action):
		self.checkBound(action)
		if self.robot.step():
			observation = self.getObservation()
			reward = self.calculateReward()
			done = self.checkTermination()
		else:
			raise SystemError("Step not executed!")
		return observation, reward, done, {}

	def render(self):
		pass

	def getObservation(self):
		observation = []
		observation.extend(self.getMotorAngles().tolist())
		observation.extend(self.getMotorVelocities().tolist())
		observation.extend(self.getMotorTorques().tolist())
		observation.extend(self.getHeadPose().tolist())
		self.observation = observation
		return self.observation

	def getObservationDimension(self):
		return len(self.getObservation())

	def _get_observation_upper_bound(self):
		upper_bound = np.zeros(self.getObservationDimension())
		upper_bound[0:self.NUM_MOTORS] = np.pi
		upper_bound[self.NUM_MOTORS:2*self.NUM_MOTORS] = self.MOTOR_SPEED_LIMIT
		upper_bound[2*self.NUM_MOTORS:3*self.NUM_MOTORS] = self.OBSERVATION_TORQUE_LIMIT
		upper_bound[3*self.NUM_MOTORS:] = np.array([0,0,0,1,0,0,0])
		return upper_bound

	def _get_observation_lower_bound(Self):
		return -self._get_observation_upper_bound()

	def defObservation(self):
		# Define observation of SnakeGym environment.
			# joint position						(n x 1)
			# joint velocity						(n x 1)
			# joint torques							(n x 1)
			# position, orientation of snake head 	(7 x 1)
			# Total:								(3n+7 x 1)
		observation_high = self._get_observation_upper_bound()
		observation_low = self._get_observation_lower_bound()
		self.observation_space = spaces.Box(observation_low, observation_high)

	def defAction(self):
		# Define actions for SnakeGym environment.
			# joint positions						(n x 1)
		action_dim = self.NUM_MOTORS
		action_high = np.array([self._action_bound] * action_dim)
		self.action_space = spaces.Box(-action_high, action_high)

	def checkBound(self, action):
		for idx, act in enumerate(actions):
			if act<-1 or act>1:
				raise ValueError("Illegal action {} at idx {}".format(act, idx))
		return True

	def calculateReward(self):
		pass

	def checkTermination(self):
		pass