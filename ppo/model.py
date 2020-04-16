import math
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def init_weights(m):
	if isinstance(m, nn.Linear):
		nn.init.normal_(m.weight, mean=0., std=0.1)
		nn.init.constant_(m.bias, 0.1)
		

class ActorCritic(nn.Module):
	def __init__(self, num_inputs, num_outputs, hidden_size):
		super(ActorCritic, self).__init__()
		
		self.critic = nn.Sequential(
			nn.Linear(num_inputs, hidden_size[0]),
			nn.ReLU(),
			nn.Linear(hidden_size[0], hidden_size[1]),
			nn.ReLU(),
			nn.Linear(hidden_size[1], 1)
		)
		
		self.actor = nn.Sequential(
			nn.Linear(num_inputs, hidden_size[0]),
			nn.ReLU(),
			nn.Linear(hidden_size[0], hidden_size[1]),
			nn.ReLU()
		)
		self.mu = nn.Linear(hidden_size[1], num_outputs)
		self.sigma = nn.Sequential(nn.Linear(hidden_size[1], num_outputs), nn.Tanh())
		
		self.apply(init_weights)
		
	def forward(self, x):
		value = self.critic(x)
		hidden_state = self.actor(x)
		mu 	  = torch.tanh(self.mu(hidden_state))
		sigma = self.sigma(hidden_state) + 0.001
		dist  = Normal(mu, sigma)
		return dist, value