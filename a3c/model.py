import math
import torch
from torch.distributions import Normal

class Network(torch.nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size=200, std=0.0):
		super(Network, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_size = hidden_size

		self.critic = torch.nn.Sequential(
			torch.nn.Linear(self.state_dim, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, 1)
		)
		
		self.actor = torch.nn.Sequential(
			torch.nn.Linear(self.state_dim, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, self.action_dim),
		)
		self.log_std = torch.nn.Parameter(torch.ones(self.action_dim) * std)
		self.apply(self.init_weights)

	@staticmethod
	def init_weights(m):
		if isinstance(m, torch.nn.Linear):
			torch.nn.init.normal_(m.weight, mean=0., std=0.1)
			torch.nn.init.constant_(m.bias, 0.1)

	def forward(self, state):
		mu = self.actor(state)
		mu = torch.tanh(mu)
		sigma = self.log_std.exp().expand_as(mu)
		values = self.critic(state)
		dist  = Normal(mu, sigma)
		return dist, values

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