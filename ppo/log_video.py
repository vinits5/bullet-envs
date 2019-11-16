# system imports
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# algorithm imports
from agent import ppo_update, compute_gae
from model import ActorCritic
from multiprocessing_env import SubprocVecEnv
import utils
from utils import print_

# environment imports
import pybullet as p
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
from SnakeGymEnv import SnakeGymEnv
import snake

class Logger:
	def __init__(self, log_dir, urdf_path=os.path.join(os.pardir, "snake/snake.urdf")):
		self.log_dir = log_dir
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")

		p.connect(p.GUI)
		self.create_env(urdf_path)
		self.net = self.create_model()
		
	def create_env(self, urdf_path):
		self.robot = snake.Snake(p, urdf_path)
		self.env = SnakeGymEnv(self.robot)

	def create_model(self):
		# Create network/policy.
		num_inputs = self.env.observation_space.shape[0]
		num_outputs = self.env.action_space.shape[0]
		hidden_size = 256
		net = ActorCritic(num_inputs, num_outputs, hidden_size).to(self.device)
		return net

	def restore_model(self, weight):
		checkpoint = torch.load(weight, map_location='cpu')
		self.net.load_state_dict(checkpoint['model'])

	def test_env(self, file_name):
		state = self.env.reset()
		done = False
		total_reward = 0
		steps = 0

		cdist = 3.0
		cyaw = 90
		cpitch = -89

		basePos = self.robot.getBasePosition()
		p.resetDebugVisualizerCamera(cameraDistance=cdist, cameraYaw=cyaw, cameraPitch=cpitch, cameraTargetPosition=basePos)

		loggingId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file_name)
		while steps < 100:
			state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
			dist, _ = self.net(state)
			next_state, reward, done, _ = self.env.step(dist.sample().cpu().numpy()[0]) #Hack
			print("Step No: {:3d}, Reward: {:2.3f} and done: {}".format(steps, reward, done))
			state = next_state
			total_reward += reward
			steps += 1
		p.stopStateLogging(loggingId)

	def log_all_videos(self):
		weights = os.listdir(os.path.join(log_dir, 'models'))
		files = [os.path.join(log_dir, 'models', w[:-4]+'.mp4') for w in weights]
		weights = [os.path.join(log_dir, 'models', w) for w in weights]
		for w, f in zip(weights, files):
			self.restore_model(w)
			self.test_env(f)

	def log_video(weight, file_name):
		self.restore_model(weight)
		self.test_env(file_name)


if __name__ == '__main__':
	log_dir = 'log'
	logger = Logger(log_dir)
	logger.log_all_videos()