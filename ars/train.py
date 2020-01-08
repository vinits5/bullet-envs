import gym
import numpy as np
import argparse
# import pybullet_envs
from gym import wrappers
import os
from tensorboardX import SummaryWriter

import pybullet as p
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
from SnakeGymEnv import SnakeGymEnv
import snake
from datetime import datetime

#################### Environment and Agent ####################
def create_env(p, args):
	urdf_path = os.path.join(BASE_DIR, os.pardir, "snake/snake.urdf")
	robot = snake.Snake(p, urdf_path, args=args)
	return SnakeGymEnv(robot, args=args)

def policy(state, weights):
	return np.matmul(weights, state.reshape(-1,1))

def test_env(env, policy, weights, normalizer=None, eval_policy=False):
	# Argument:
		# env:			Object of the gym environment.
		# policy:		A function that will take weights, state and returns actions
	state = env.reset()
	state = state + np.random.random_sample(state.shape)#*0.01
	done = False
	total_reward = 0.0
	total_states = []
	steps = 0

	while not done and steps<200:
		if normalizer:
			if not eval_policy: normalizer.observe(state)
			state = normalizer.normalize(state)
		action = policy(state, weights)
		next_state, reward, done, _ = env.step(action)

		# # Trick to avoid local optima.
		# if abs(next_state[2]) < 0.001:
		# 	reward = -100
		# 	done = True

		total_states.append(state)
		total_reward += reward
		steps += 1
		state = next_state
	if eval_policy: return float(total_reward), steps
	else: return float(total_reward)

#################### ARS algorithm ####################
def sort_directions(data, b):
	reward_p, reward_n = data
	reward_max = []
	for rp, rn in zip(reward_p, reward_n):
		reward_max.append(max(rp, rn))

	# ipdb.set_trace()
	idx = np.argsort(reward_max)	# Sort rewards and get indices.
	idx = np.flip(idx)				# Flip to get descending order.

	return idx

def update_weights(data, lr, b, weights):
	reward_p, reward_n, delta = data
	idx = sort_directions([reward_p, reward_n], b)

	step = np.zeros(weights.shape)
	for i in range(b):
		step += [reward_p[idx[i]] - reward_n[idx[i]]]*delta[idx[i]]

	sigmaR = np.std(np.array(reward_p)[idx][:b] + np.array(reward_n)[idx][:b])
	weights += (lr*1.0)/(b*sigmaR*1.0)*step

	return weights

def sample_delta_normal(size):
	return np.random.normal(size=size)

def sample_delta(size):
	return np.random.randn(*size)

#################### Normalizing the states #################### 
class Normalizer():
	def __init__(self, nb_inputs):
		self.n = np.zeros(nb_inputs)
		self.mean = np.zeros(nb_inputs)
		self.mean_diff = np.zeros(nb_inputs)
		self.var = np.zeros(nb_inputs)

	def observe(self, x):
		self.n += 1.
		last_mean = self.mean.copy()
		self.mean += (x - self.mean) / self.n
		self.mean_diff += (x - last_mean) * (x - self.mean)
		self.var = (self.mean_diff / self.n).clip(min=1e-2)

	def normalize(self, inputs):
		obs_mean = self.mean
		obs_std = np.sqrt(self.var)
		return (inputs - obs_mean) / obs_std

	def store(self, path):
		np.savetxt(os.path.join(path, 'mean.txt'), self.mean)
		np.savetxt(os.path.join(path, 'var.txt'), self.var)

#################### Training ARS Class #################### 
class ARS:
	def __init__(self, args):
		self.v = args.v
		self.N = args.N
		self.b = args.b
		self.lr = args.lr
		self.args = args

		if not os.path.exists(args.log): 
			os.mkdir(args.log)
			os.mkdir(os.path.join(args.log, 'models'))
			os.mkdir(os.path.join(args.log, 'videos'))
			os.system('cp train.py %s'%(args.log))

		p.connect(p.DIRECT)
		self.env = create_env(p, args)
		# self.env = wrappers.Monitor(self.env, os.path.join(args.log,'videos'), force=True)

		self.size = [self.env.action_space.shape[0], self.env.observation_space.shape[0]]
		self.weights = np.zeros(self.size)
		if args.normalizer: self.normalizer = Normalizer([1,self.size[1]])
		else: self.normalizer=None

	def save_policy(self, counter):
		path = os.path.join(self.args.log, 'models', 'policy'+str(counter))
		if not os.path.exists(path): os.mkdir(path)
		np.savetxt(os.path.join(path, 'weights.txt'), self.weights)
		self.normalizer.store(path)

	def train_one_epoch(self):
		delta = [sample_delta(self.size) for _ in range(self.N)]

		reward_p = [test_env(self.env, policy, self.weights + self.v*x, normalizer=self.normalizer) for x in delta]
		reward_n = [test_env(self.env, policy, self.weights - self.v*x, normalizer=self.normalizer) for x in delta]
		
		return update_weights([reward_p, reward_n, delta], self.lr, self.b, self.weights)

	def train(self):
		writer = SummaryWriter(self.args.log)
		print('Training Begins!')
		counter = 0

		while counter < 10000:
			print('Counter: {}'.format(counter))
			self.weights = self.train_one_epoch()

			test_reward, num_plays = test_env(self.env, policy, self.weights, normalizer=self.normalizer, eval_policy=True)
			self.save_policy(counter)
			writer.add_scalar('test_reward', test_reward, counter)
			writer.add_scalar('episodic_steps', num_plays, counter)
			print('Iteration: {} and Reward: {}'.format(counter, test_reward))
			counter += 1

		counter = 0		
		while True:
			print(test_env(self.args, policy, self.weights, normalizer=self.normalizer))
			counter += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='ARS Parameters')
	parser.add_argument('--v', type=float, default=0.03, help='noise in delta')
	parser.add_argument('--N', type=int, default=16, help='No of perturbations')
	parser.add_argument('--b', type=int, default=16, help='No of top performing directions')
	parser.add_argument('--lr', type=float, default=0.02, help='Learning Rate')
	parser.add_argument('--normalizer', type=bool, default=True, help='use normalizer')
	parser.add_argument('--log', type=str, default='exp_snake', help='Log folder to store videos')

	parser.add_argument('--mode', type=str, default='train', help='Options: [train/test]')

	parser.add_argument('--alpha', type=float, default=1.0, help='Weight applied to forward motion reward')
	parser.add_argument('--beta', type=float, default=0.01, help='Weight applied to drift reward')
	parser.add_argument('--gamma', type=float, default=0.1, help='Weight applied to energy reward')

	# snake file parameters
	parser.add_argument('--selfCollisionEnabled', type=bool, default=True, help='Collision between links')
	parser.add_argument('--motorVelocityLimit', type=float, default=np.inf, help='Joint velocity limit')
	parser.add_argument('--motorTorqueLimit', type=float, default=np.inf, help='Joint torque limit')
	parser.add_argument('--kp', type=float, default=10.0, help='Kp value for joint control')
	parser.add_argument('--kd', type=float, default=0.1, help='Kd value for joint control')
	parser.add_argument('--gaitSelection', type=int, default=1, help='Choose gait')
	parser.add_argument('--scaling_factor', type=float, default=6, help='scaling applied to actions')
	parser.add_argument('--cam_dist', type=float, default=5.0, help='Camera Distance')
	parser.add_argument('--cam_yaw', type=float, default=50, help='Camera Yaw')
	parser.add_argument('--cam_pitch', type=float, default=-35, help='Camera Pitch')
	parser.add_argument('--cam_roll', type=float, default=0, help='Camera Roll')
	parser.add_argument('--upAxisIndex', type=int, default=2, help='Camera Axis')
	parser.add_argument('--render_height', type=int, default=720, help='Height of camera image')
	parser.add_argument('--render_width', type=int, default=960, help='Width of camera image')
	parser.add_argument('--fov', type=int, default=60, help='Field of view for camera')
	parser.add_argument('--nearVal', type=float, default=0.1, help='minimum clipping value for camera')
	parser.add_argument('--farVal', type=float, default=100, help='maximum clipping value for camera')

	args = parser.parse_args()
	args.log = os.path.join(args.log)
	ars = ARS(args)
	ars.train()