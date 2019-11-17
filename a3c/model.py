import math
import torch
import torch.nn as nn
from torch.distributions import Normal

class Network(torch.nn.Module):
	def __init__(self, num_inputs, num_outputs, hidden_size=[256,256]):
		super(Network, self).__init__()
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
		self.sigma = nn.Sequential(nn.Linear(hidden_size[1], num_outputs), nn.Sigmoid())
		self.apply(self.init_weights)

	@staticmethod
	def init_weights(m):
		if isinstance(m, torch.nn.Linear):
			torch.nn.init.normal_(m.weight, mean=0., std=0.1)
			torch.nn.init.constant_(m.bias, 0.1)

	def forward(self, state):
		value = self.critic(state)
		hidden_state = self.actor(state)
		mu 	  = torch.tanh(self.mu(hidden_state))
		sigma = self.sigma(hidden_state) + 0.001
		dist  = Normal(mu, sigma)
		return dist, value

	def loss_function(self, state, action, values_t):
		self.train()
		dist, values = self.forward(state)
		td_error = values_t - values
		critic_loss = td_error.pow(2)

		log_prob = dist.log_prob(action)
		entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(dist.scale)  # exploration
		actor_loss = -(log_prob * td_error.detach() + 0.005 * entropy)
		total_loss = (actor_loss + critic_loss).mean()
		return total_loss

	def choose_action(self, state):
		self.training = False
		dist, _ = self.forward(state)
		return dist.sample().numpy()

if __name__ == '__main__':
	net = Network(3,1)
	state = torch.Tensor([0.3, 0.2, 0.1])
	action = torch.Tensor([0.5])