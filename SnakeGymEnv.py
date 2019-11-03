import gym
import numpy as np

class SnakeGymEnv(gym.Env):
	def __init__(self, robot):
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
			self._observation = observation
			# done = self.checkTermination()
			done = False
		else:
			raise SystemError("Action not executed!")
		return observation, reward, done, {}

	def render(self):
		pass

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
		action_dim = self.robot.numMotors
		action_high = np.array([self._action_bound] * action_dim)
		self.action_space = gym.spaces.Box(-action_high, action_high)

	# Check if the network output is in feasible range.
	def checkBound(self, action):
		for idx, act in enumerate(action):
			if act<-1 or act>1:
				raise ValueError("Illegal action {} at idx {}".format(act, idx))
		return True

	def calculateReward(self, observation):
		print(observation[24], self._observation[24])
		return observation[24] - self._observation[24]

	def checkTermination(self):
		pass