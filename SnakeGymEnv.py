import gym
import numpy as np

class SnakeGymEnv(gym.Env):
	def __init__(self, robot):
		print("Snake Gym environment Created!")
		self.robot = robot
		self._action_bound = 1
		self.mode = "rgb_array"

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
			self._observation = observation
		else:
			raise SystemError("Action not executed!")
		return observation, reward, done, {}

	def render(self):
		if self.mode != "rgb_array":
			return np.array([])
		return self.robot.render()

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
		action_dim = int(self.robot.numMotors/2)
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
		total_reward = reward_xMotion
		return total_reward

	def checkTermination(self, observation):
		if abs(observation[49]) > 0.5: return True
		else: return False