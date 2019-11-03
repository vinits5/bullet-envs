import gym
import numpy as np

class SnakeGymEnv(gym.Env):
	def __init__(self, robot):
		self.robot = robot
		self._action_bound = 1
		self._observation = []

		self.robot.buildMotorList()
		self.defObservationSpace()
		self.defActionSpace()

	def reset(self, reloadURDF):
		assert self.robot.reset(reloadURDF), "Error in reset!"
		return self.robot.getObservation()

	def step(self, action):
		self.checkBound(action)
		if self.robot.step(action):
			observation = self.robot.getObservation()
			# reward = self.calculateReward()
			# done = self.checkTermination()
			reward = 0
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

	def calculateReward(self):
		pass

	def checkTermination(self):
		pass