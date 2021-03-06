import gym
import numpy as np

class SnakeGymEnv(gym.Env):
	def __init__(self, robot, args=None):
		print("Snake Gym environment Created!")
		if args is not None:
			self.alpha 				= args.alpha
			self.beta 				= args.beta
			self.gamma 				= args.gamma
			self.mode 				= args.mode
			self._gaitSelection 	= args.gaitSelection
		else:
			self.alpha 				= 1
			self.beta 				= 0.01
			self.gamma 				= 0.1
			self.mode				= 'train'
			self._gaitSelection 	= 1

		self.robot = robot
		self._action_bound = 1

		self.robot.reset(hardReset=True)
		self.robot.buildMotorList()
		self.defObservationSpace()
		self.defActionSpace()

	def reset(self, hardReset=False):
		assert self.robot.reset(hardReset=hardReset), "Error in reset!"
		self._observation = self.robot.getObservation()
		return self._observation

	def step(self, action):
		self.checkBound(action)
		if self.robot.step(action):
			observation = self.robot.getObservation()
			reward = self.calculateReward(observation)
			done = self.checkTermination(observation)
			if done:
				reward += (-5)
				self.reset()
			self._observation = observation
			if self.mode == 'test':
				info = {'frames':self.robot.imgs, 'internal_observations':self.robot.step_internal_observations, 'link_positions': self.robot.link_positions}
			else:
				info = {}

		else:
			raise SystemError("Action not executed!")
		return observation, reward, done, info

	def render(self):
		if self.mode == 'train':
			return np.array([])
		elif self.mode == 'test':
			return self.robot.render()
		else:
			return np.array([])

	def defObservationSpace(self):
		# Define observation of SnakeGym environment.
			# joint position						(n x 1)
			# joint velocity						(n x 1)
			# joint torques							(n x 1)
			# position, orientation of snake head 	(7 x 1)
			# Total:								(3n+7 x 1)
		observation_high = self.robot.getObservationUpperBound()
		observation_low = self.robot.getObservationLowerBound()
		self.observation_space = gym.spaces.Box(observation_low, observation_high)

	def defActionSpace(self):
		# Define actions for SnakeGym environment.
			# joint positions						(n x 1)
		if self._gaitSelection == 0 or self._gaitSelection == 1:
			action_dim = int(self.robot.numMotors/2)
		else: action_dim = int(self.robot.numMotors)
		
		action_high = np.array([self._action_bound] * action_dim)
		self.action_space = gym.spaces.Box(-action_high, action_high)

	# Check if the network output is in feasible range.
	def checkBound(self, action):
		for idx, act in enumerate(action):
			if act<-1 or act>1:
				# raise ValueError("Illegal action {} at idx {}".format(act, idx))
				# print('Clipped')
				action[idx] = np.clip(act, -1, 1)
		return True

	def calculateReward(self, observation):
		reward_xMotion = observation[48] - self._observation[48]
		reward_yMotion = abs(observation[49] - self.robot.START_POSITION[2])
		reward_energy = self.robot.calculateEnergy(observation)
		if np.abs(observation[55]) > 10: reward_collision = -10
		else: reward_collision = 0
		total_reward = self.alpha*reward_xMotion + reward_collision - self.beta*reward_yMotion- self.gamma*reward_energy
		return total_reward

	def checkTermination(self, observation):
		if abs(observation[9]) > 0.5: return True
		elif self.robot.checkSnakeHeight(): return True
		elif self.robot.endDue2Height == True: return True
		else: return False