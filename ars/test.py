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
	# state: 	MountainCar (2,1)
	# weights:	MountainCar (1,2)
	return np.matmul(weights, state.reshape(-1,1))#.reshape(-1,1)

def test_env(env, policy, weights, normalizer=None, path=None):
	# Argument:
		# env:			Object of the gym environment.
		# policy:		A function that will take weights, state and returns actions
	if path: np.savetxt(path+'.txt', weights)		
	state = env.reset()
	state = state + np.random.random_sample(state.shape)#*0.01
	done = False
	total_reward = 0.0
	total_states = []
	steps = 0

	while not done and steps<5000:
		if normalizer:
			state = normalizer.normalize(state)
		action = policy(state, weights)
		next_state, reward, done, _ = env.step(action)
		#if abs(next_state[2]) < 0.0001*10:
		#	reward = -100
		#	done = True
		print(next_state[2], reward, done)
		# reward = max(min(reward, 1), -1)
		env.render()	

		total_states.append(state)
		total_reward += reward
		steps += 1
		state = next_state
	print(float(total_reward), steps)
	if path is None: return float(total_reward)
	else: return float(total_reward), steps

#################### Normalizing the states #################### 
class Normalizer():
	def __init__(self, nb_inputs):
		self.mean = np.zeros(nb_inputs)
		self.var = np.zeros(nb_inputs)

	def restore(self, path):
		self.mean = np.loadtxt(os.path.join(path, 'mean.txt'))
		self.var = np.loadtxt(os.path.join(path, 'var.txt'))

	def normalize(self, inputs):
		obs_mean = self.mean
		obs_std = np.sqrt(self.var)
		return (inputs - obs_mean) / obs_std


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

	idx = 59
	path = os.path.join('exp_snake', 'models', 'policy'+str(idx))
	weights = np.loadtxt(os.path.join(path, 'weights.txt'))
	args = parser.parse_args()

	p.connect(p.GUI)
	env = create_env(p, args)
	#env = wrappers.Monitor(env, 'videos', force=True)
	normalizer = Normalizer([1, env.observation_space.shape[0]])
	normalizer.restore(path)

	test_env(env, policy, weights, normalizer=normalizer)