import torch
import numpy as np
import torch.multiprocessing as mp
import gym
import model

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
from SnakeGymEnv import SnakeGymEnv
import snake

UPDATE_GLOBAL_ITER = 5
MAX_EP = 4000
MAX_EPISODE_STEPS = 200
GAMMA = 0.9

def v_wrap(np_array, dtype=np.float32):
	if np_array.dtype != dtype:
		np_array = np_array.astype(dtype)
	return torch.from_numpy(np_array)

def push_and_pull(optimizer, local_network, global_network, done, next_state, buffer_states, buffer_actions, buffer_rewards, gamma):
	if done:
		next_state_value = 0.0               # terminal
	else:
		next_state_value = local_network.forward(v_wrap(next_state[None, :]))[-1].data.numpy()[0, 0]

	buffer_value_target = []
	for r in buffer_rewards[::-1]:    # reverse buffer r
		next_state_value = r + gamma * next_state_value
		buffer_value_target.append(next_state_value)
	buffer_value_target.reverse()

	loss = local_network.loss_function(
		v_wrap(np.vstack(buffer_states)),
		v_wrap(np.array(buffer_actions), dtype=np.int32) if buffer_actions[0].dtype == np.int64 else v_wrap(np.vstack(buffer_actions)),
		v_wrap(np.array(buffer_value_target)[:, None]))

	# calculate local gradients and push local parameters to global
	optimizer.zero_grad()
	loss.backward()
	for lp, gp in zip(local_network.parameters(), global_network.parameters()):
		gp._grad = lp.grad
	optimizer.step()

	# pull global parameters
	local_network.load_state_dict(global_network.state_dict())

def record(global_ep, global_ep_r, episode_reward, res_queue, name):
	with global_ep.get_lock():
		global_ep.value += 1
	with global_ep_r.get_lock():
		if global_ep_r.value == 0.:
			global_ep_r.value = episode_reward
		else:
			global_ep_r.value = global_ep_r.value * 0.99 + episode_reward * 0.01
	res_queue.put(global_ep_r.value)
	print(name, "Ep:", global_ep.value, "| Ep_r: %.4f" % global_ep_r.value)


class SharedAdam(torch.optim.Adam):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8, weight_decay=0):
		super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
		# State initialization
		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				state['step'] = 0
				state['exp_avg'] = torch.zeros_like(p.data)
				state['exp_avg_sq'] = torch.zeros_like(p.data)

				# share in memory
				state['exp_avg'].share_memory_()
				state['exp_avg_sq'].share_memory_()


class Worker(mp.Process):
	def __init__(self, global_network, optimizer, global_ep, global_ep_r, res_queue, worker_name, pybullet_client, urdf_path):
		super(Worker, self).__init__()
		self.device = 'cpu'
		if torch.cuda.is_available(): self.device = 'cuda'
		self.worker_name = 'worker_%i'%worker_name
		self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue

		self.global_network = global_network
		self.optimizer = optimizer
		# self.env = gym.make('Pendulum-v0').unwrapped
		robot = snake.Snake(pybullet_client, urdf_path)
		self.env = SnakeGymEnv(robot)

		self.local_network = model.Network(self.env.observation_space.shape[0], self.env.action_space.shape[0])
		# self.local_optimizer = torch.optim.Adam(lr=lr)

	def run(self):
		total_step = 1
		while self.g_ep.value < MAX_EP:
			state = self.env.reset()
			buffer_states, buffer_actions, buffer_rewards = [], [], []
			episode_reward = 0.0

			for t in range(MAX_EPISODE_STEPS):
				state = torch.FloatTensor(state).to(self.device)
				action = self.local_network.choose_action(state)
				next_state, reward, done, _ = self.env.step(action)

				if t == MAX_EPISODE_STEPS - 1:
					done = True

				episode_reward += reward
				buffer_actions.append(action)
				buffer_states.append(state)
				buffer_rewards.append(reward)    # normalize

				if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
					# sync
					push_and_pull(self.optimizer, self.local_network, self.global_network, done, next_state, buffer_states, buffer_actions, buffer_rewards, GAMMA)
					buffer_states, buffer_actions, buffer_rewards = [], [], []

					# if done:  # done and print information
					record(self.g_ep, self.g_ep_r, episode_reward, self.res_queue, self.worker_name)
						# break
				state = next_state
				total_step += 1

		self.res_queue.put(None)

# if __name__ == '__main__':
# 	print("Cannot test easily!!")