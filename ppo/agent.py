import math
import random

import math
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
	values = values + [next_value]
	gae = 0
	returns = []
	for step in reversed(range(len(rewards))):
		delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
		gae = delta + gamma * tau * masks[step] * gae
		returns.insert(0, gae + values[step])
	return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
	batch_size = states.size(0)
	for _ in range(batch_size // mini_batch_size):
		rand_ids = np.random.randint(0, batch_size, mini_batch_size)
		yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, writer, frame_idx, clip_param=0.2):
	total_loss = 0.0
	# total_advantage = 0.0
	total_actor_loss = 0.0
	total_critic_loss = 0.0
	total_entropy = 0.0
	for _ in range(ppo_epochs):
		for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
			dist, value = model(state)
			entropy = dist.entropy().mean()
			new_log_probs = dist.log_prob(action)

			ratio = (new_log_probs - old_log_probs).exp()
			surr1 = ratio * advantage
			surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

			actor_loss  = - torch.min(surr1, surr2).mean()
			critic_loss = (return_ - value).pow(2).mean()

			loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
			total_loss += loss.item()
			total_actor_loss += actor_loss.item()
			total_critic_loss += critic_loss.item()
			total_entropy += entropy.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	total_loss = total_loss/(ppo_epochs*1.0)
	total_loss = total_loss/(states.size(0)/mini_batch_size)

	total_critic_loss = total_critic_loss/(ppo_epochs*1.0)
	total_critic_loss = total_critic_loss/(states.size(0)/mini_batch_size)

	total_actor_loss = total_actor_loss/(ppo_epochs*1.0)
	total_actor_loss = total_actor_loss/(states.size(0)/mini_batch_size)

	total_entropy = total_entropy/(ppo_epochs*1.0)
	total_entropy = total_entropy/(states.size(0)/mini_batch_size)

	writer.add_scalar('loss/epoch', total_loss, frame_idx)
	writer.add_scalar('critic_loss/epoch',total_critic_loss,frame_idx)
	writer.add_scalar('actor_loss/epoch',total_actor_loss,frame_idx)
	writer.add_scalar('entropy/epoch',total_entropy,frame_idx)
