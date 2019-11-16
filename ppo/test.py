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

# Parameters
urdf_path = os.path.join(os.pardir, "snake/snake.urdf")
hidden_size = [256,256]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
log_dir = 'log_13_11_2019_18_03_56'

# Create test environment.
p.connect(p.GUI)
robot = snake.Snake(p, urdf_path)
env = SnakeGymEnv(robot)

# Check availability of cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# State space and action space
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

# Create network/policy.
net = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)

checkpoint = torch.load(os.path.join(log_dir,'weights.pth'), map_location='cpu')
net.load_state_dict(checkpoint['model'])

state = env.reset()

done = False
total_reward = 0
steps = 0
print_('Test Started...', color='r', style='bold')
while steps < 100:
	state = torch.FloatTensor(state).unsqueeze(0).to(device)
	dist, _ = net(state)
	next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0]) #Hack
	print("Step No: {:3d}, Reward: {:2.3f} and done: {}".format(steps, reward, done))
	state = next_state
	total_reward += reward
	steps += 1
print_('Total Reward: {}'.format(total_reward), color='bl', style='bold')
print('Test Ended!')