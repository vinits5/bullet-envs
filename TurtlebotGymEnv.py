import gym
import numpy as np
from turtlebot import Turtlebot

class TurtlebotGymEnv(gym.Env):
	def __init__(self, robot, args=None):
		print("Turtlebot Gym environment Created!")
		if args is not None:
			self.mode 				= args.mode
		else:
			self.mode				= 'train'

		self.robot = robot
		self._action_bound_high = 1
		self._action_bound_low = 0
		self.discrete = args.discrete

		self.robot.reset(hardReset=True)
		self.robot.buildMotorList()
		self.defObservationSpace()
		self.defActionSpace()

	def reset(self, hardReset=False):
		assert self.robot.reset(hardReset=hardReset), "Error in reset!"
		self._observation = self.robot.getObservation()
		return self._observation

	def step(self, action):
		if not self.discrete: action = self.checkBound(action)			# only for cotinuous
		if self.robot.step(action):
			observation = self.robot.getObservation()
			# print(observation[126:])
			reward, done_reward = self.calculateReward(observation,action)
			# reward = 0.0
			done = self.checkTermination(observation)
			# print(self.robot.goal,done_reward,done,reward)
			if (done_reward or done):
				if done_reward: self.reset(hardReset=True)
				else: self.reset()
			self._observation = observation
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
		# Define observation of Turtlebot environment.
			# lidar distances						(n-directions x 1)
			# Total:								(n-directions x 1)
		observation_high = self.robot.getObservationUpperBound()
		observation_low = self.robot.getObservationLowerBound()
		self.observation_space = gym.spaces.Box(observation_low, observation_high)

	def defActionSpace(self):
		# Define actions for Turtlebot environment.
			# joint positions						(n x 1)
		if self.discrete:
			# For discrete actions.
			self.action_space = gym.spaces.Discrete(self.robot.action_dim)
		else:
			# For continuous actions.
			action_dim = int(self.robot.numMotors)
			action_high = np.array([self._action_bound_high] * action_dim)
			action_low = np.array([self._action_bound_low] * action_dim)
			self.action_space = gym.spaces.Box(action_low, action_high)

	# Check if the network output is in feasible range.
	def checkBound(self, action):
		action = np.array(action)
		action_high = action < self.action_space.high
		action_low = action > self.action_space.low
		idx_high = np.where(action_high == False)
		idx_low = np.where(action_low == False)

		action[idx_high[0]] = self.action_space.high[0]
		action[idx_low[0]] = self.action_space.low[0]
		return action

	def calculateReward(self, observation, action):
		done = False
		basePosition = self.robot.getBasePosition()
		goalPosition = self.robot.goal
		distance_reward = ((basePosition[0] - goalPosition[0])**2 + (basePosition[1] - goalPosition[1])**2)**0.5
		lidar_space = len(observation)-2
		if distance_reward < self.robot.goal_reached_threshold:
			goal_achieved_reward = 100
			done = True
		else:
			goal_achieved_reward = 0

		distance_reward = distance_reward / ((goalPosition[0]**2 + goalPosition[1]**2)**0.5)

		rotation_velocity = (action[1]-action[0])/2.0
		forward_velocity = (action[1]+action[0])/2.0
		rotation_reward = abs(rotation_velocity)*10
		# forward_velocity_reward = forward_velocity*2.0
		# rotation_reward = 0.0
		forward_velocity_reward = 0.0

		# print("rotation reward",forward_velocity_reward)

		step_reward = -1
		if min(observation[0:lidar_space])<0.2:
			collision_reward = -100.0
		else:
			collision_reward = 0
		# print(distance_reward,step_reward,collision_reward,goal_achieved_reward, rotation_reward,forward_velocity_reward)
		return (-distance_reward + step_reward + collision_reward + goal_achieved_reward- rotation_reward + forward_velocity_reward, done)
		# return (step_reward + collision_reward + goal_achieved_reward, done)

	def checkTermination(self, observation):
		done = False
		lidar_space = len(observation)-2
		# print(min(observation[0:126]))
		if min(observation[0:lidar_space])<0.2	:
			# print("collison detected",min(observation[0]))
			done = True
		return done