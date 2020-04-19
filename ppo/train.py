# system imports
import gym
import pdb
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
from params import params
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
from datetime import datetime

def save_checkpoint(state, filename):
	torch.save(state, '{}'.format(filename))

def train(args):
	# hyper-params:
	frame_idx  		 = 0
	hidden_size      = args.hidden_size
	lr               = args.lr
	num_steps        = args.num_steps
	mini_batch_size  = args.mini_batch_size
	ppo_epochs       = args.ppo_epochs
	threshold_reward = args.threshold_reward
	max_frames 		 = args.max_frames
	# test_rewards 	 = []
	num_envs 		 = args.num_envs
	test_epochs		 = args.test_epochs
	resume_training	 = args.resume_training
	best_test_reward = 0.0
	urdf_path		 = os.path.join(BASE_DIR, os.pardir, "turtlebot_urdf/turtlebot.urdf")
	log_dir 		 = args.log_dir

	now = datetime.now()
	log_dir = log_dir + '_' + now.strftime('%d_%m_%Y_%H_%M_%S')

	# Check cuda availability.
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")


	p.connect(p.DIRECT)
	writer = SummaryWriter(log_dir)

	# Create training log.
	textio = utils.IOStream(os.path.join(log_dir, 'train.log'), args=args)
	# textio.log_params(device, num_envs, lr, threshold_reward)	
	utils.logFiles(log_dir)

	# create multiple environments.
	envs = [utils.make_env(p, urdf_path, args=args) for i in range(num_envs)]
	envs = SubprocVecEnv(envs)

	# pdb.set_trace()	# Debug
	num_inputs = envs.observation_space.shape[0]
	if not args.discrete: num_outputs = envs.action_space.shape[0]		# for continuous actions
	else: num_outputs = envs.action_space.n 				# for discrete actions

	# Create Policy/Network
	net = ActorCritic(num_inputs, num_outputs, hidden_size, args.discrete).to(device)
	optimizer = optim.Adam(net.parameters(), lr=lr)

	# If use pretrained policy.
	if resume_training:
		if os.path.exists(resume_training):
			checkpoint = torch.load(resume_training)
			frame_idx = checkpoint['frame_idx']
			net.load_state_dict(checkpoint['model'])
			best_test_reward = checkpoint['best_test_reward']

	# Initial Reset for Environment.
	state = envs.reset()
	early_stop = False

	# Create env for policy testing.
	robot = turtlebot.Turtlebot(p, urdf_path, args=args)
	env = TurtlebotGymEnv(robot, args=args)

	print_('\nTraining Begins ...', color='r', style='bold')
	textio.log('Training Begins ...')
	while frame_idx < max_frames and not early_stop:
		print_('\nTraining Policy!', color='r', style='bold')
		textio.log('\n############## Epoch: %0.5d ##############'%(int(frame_idx/num_steps)))

		# Memory buffers
		log_probs = []
		values    = []
		states    = []
		actions   = []
		rewards   = []
		masks     = []
		entropy   = 0
		total_reward = 0.0

		for i in range(num_steps):
			print('Epoch: [{}/{}] & Steps taken: [{}/{}]\r'.format(int(frame_idx/num_steps)+1, int(max_frames/num_steps), i+1, num_steps), end="")
			state = torch.FloatTensor(state).to(device)

			# Find action using policy.
			dist, value = net(state)
			action = dist.sample()

			# Take actions and find MDP.
			next_state, reward, done, _ = envs.step(action.cpu().numpy())
			total_reward += sum(reward)
			textio.log('Epoch: {} and Reward: {}'.format(int(frame_idx%num_steps), total_reward))

			# Calculate log(policy)
			log_prob = dist.log_prob(action)
			entropy += dist.entropy().mean()

			# Create Experiences
			log_probs.append(log_prob)
			values.append(value)
			rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
			masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
			states.append(state)
			actions.append(action)
			
			# Update state.
			state = next_state
			frame_idx += 1

			# Test Trained Policy.
			if frame_idx % 100 == 0:
				print_('\n\nEvaluate Policy!', color='bl', style='bold')
				test_reward = np.mean([utils.test_env(env, net, test_idx) for test_idx in range(test_epochs)])

				# test_rewards.append(test_reward)
				# utils.plot(frame_idx, test_rewards)	# not required due to tensorboardX.
				writer.add_scalar('test_reward', test_reward, frame_idx)
				
				print_('\nTest Reward: {}\n'.format(test_reward), color='bl', style='bold')
				textio.log('\nTest Reward: {}'.format(test_reward))

				# Save various factors of training.
				snap = {'frame_idx': frame_idx,
						'model': net.state_dict(),
						'best_test_reward': best_test_reward,
						'optimizer' : optimizer.state_dict()}

				if best_test_reward < test_reward:
					save_checkpoint(snap, os.path.join(log_dir, 'weights_bestPolicy.pth'))
					best_test_reward = test_reward
				save_checkpoint(snap, os.path.join(log_dir,'weights.pth'))
				if test_reward > threshold_reward: early_stop = True
			if frame_idx % 1000 == 0:
				if not os.path.exists(os.path.join(log_dir, 'models')): os.mkdir(os.path.join(log_dir, 'models'))
				save_checkpoint(snap, os.path.join(log_dir, 'models', 'weights_%0.5d.pth'%frame_idx))

				
		# Calculate Returns
		next_state = torch.FloatTensor(next_state).to(device)
		_, next_value = net(next_state)
		returns = compute_gae(next_value, rewards, masks, values)

		# Concatenate experiences for multiple environments.
		returns   = torch.cat(returns).detach()
		log_probs = torch.cat(log_probs).detach()
		values    = torch.cat(values).detach()
		states    = torch.cat(states)
		actions   = torch.cat(actions)
		advantage = returns - values
		
		writer.add_scalar('reward/episode', total_reward, frame_idx)
		textio.log('Total Training Reward: {}'.format(total_reward))

		# Update the Policy.
		ppo_update(net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, writer, frame_idx, args.discrete)
		envs.reset()


if __name__ == '__main__':
	args = params()
	train(args)