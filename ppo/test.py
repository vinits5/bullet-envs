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

# log videos
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

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
from TurtlebotGymEnv import TurtlebotGymEnv
import turtlebot
from params import params


def log_video(frames):
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
	for frame in frames:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		out.write(frame)
	out.release()

def running_test(log_dir, max_steps=100, create_video=False):
	args = params()
	urdf_path = os.path.join(os.pardir, os.path.join(BASE_DIR, os.pardir, "turtlebot_urdf/turtlebot.urdf"))
	hidden_size = [64, 32]
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")


	# Create test environment.
	if create_video: p.connect(p.GUI)
	else: p.connect(p.GUI)
	cdist = 1.5
	cyaw = -30
	cpitch = -90
	# p.resetDebugVisualizerCamera(cameraDistance=cdist, cameraYaw=cyaw, cameraPitch=cpitch, cameraTargetPosition=[1.28,0,0])
	robot = turtlebot.Turtlebot(p, urdf_path, args)
	env = TurtlebotGymEnv(robot, args)

	# Check availability of cuda
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	# State space and action space
	num_inputs = env.observation_space.shape[0]
	if not args.discrete: num_outputs = env.action_space.shape[0]
	else: num_outputs = env.action_space.n

	# Create network/policy.
	net = ActorCritic(num_inputs, num_outputs, hidden_size, discrete=args.discrete).to(device)

	checkpoint = torch.load(os.path.join(log_dir,'models/weights_21000.pth'), map_location='cpu')
	net.load_state_dict(checkpoint['model'])

	if create_video: frames = []
	state = env.reset()
	print("Goal Position of Robot: ", env.robot.goal)

	if create_video: frames.append(env.render())
	# if create_video: frames.append(img)
	done = False
	total_reward = 0
	steps = 0
	print_('Test Started...', color='r', style='bold')
	STATES = []
	link_positions = []

	while steps < max_steps:
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		dist, _ = net(state)
		# print(dist.sample().cpu().numpy()[0])
		action = dist.sample().cpu().numpy()[0]
		next_state, reward, done, info = env.step(action)

		# print(info.keys())

		print("Step No: {:3d}, Reward: {:2.3f}, action: {}".format(steps, reward, action))
		state = next_state
		total_reward += reward
		steps += 1
		STATES.append(state)
	print_('Total Reward: {}'.format(total_reward), color='bl', style='bold')
	print('Test Ended!')
	if create_video: log_video(frames)
	return STATES

state = running_test('log_23_04_2020_19_21_01')
# print(state)
# print(type(state))