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

def save_checkpoint(state, filename):
	torch.save(state, '{}'.format(filename))

def train():
	# hyper-params:
	hidden_size      = 256
	lr               = 3e-4
	num_steps        = 20
	mini_batch_size  = 5
	ppo_epochs       = 4
	threshold_reward = 200
	max_frames 		 = 15000
	frame_idx  		 = 0
	test_rewards 	 = []
	urdf_path		 = os.path.join(BASE_DIR, os.pardir, "snake/snake.urdf")
	num_envs 		 = 2
	test_epochs		 = 10
	log_dir 		 = 'log'
	resume_training	 = ''
	best_test_reward = 0.0

	# Check cuda availability.
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	p.connect(p.DIRECT)
	writer = SummaryWriter(log_dir)

	# create multiple environments.
	envs = [utils.make_env(p, urdf_path) for i in range(num_envs)]
	envs = SubprocVecEnv(envs)

	# pdb.set_trace()	# Debug
	num_inputs = envs.observation_space.shape[0]
	num_outputs = envs.action_space.shape[0]

	# Create Policy/Network
	net = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
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
	robot = snake.Snake(p, urdf_path)
	env = SnakeGymEnv(robot)

	print_('\nTraining Begins ...', color='r', style='bold')
	while frame_idx < max_frames and not early_stop:
		print_('\nTraining Policy!', color='r', style='bold')

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
			print('Steps taken: {} & Epoch: {}'.format(i, int(frame_idx/20)))
			state = torch.FloatTensor(state).to(device)

			# Find action using policy.
			dist, value = net(state)
			action = dist.sample()
			action = action*0.1 #HACK

			# Take actions and find MDP.
			next_state, reward, done, _ = envs.step(action.cpu().numpy())
			total_reward += sum(reward)

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
			if frame_idx % 10 == 0:
				print_('\nEvaluate Policy!', color='bl', style='bold')
				test_reward = np.mean([utils.test_env(env, net, test_idx) for test_idx in range(test_epochs)])

				test_rewards.append(test_reward)
				# utils.plot(frame_idx, test_rewards)	# not required due to tensorboardX.
				writer.add_scalar('test_reward', test_reward, frame_idx)
				
				print_('Test Reward: {}\n'.format(test_reward), color='bl', style='bold')

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

		# Update the Policy.
		ppo_update(net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)


if __name__ == '__main__':
	train()