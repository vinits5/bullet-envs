import torch
import model
import torch.multiprocessing as mp
from agent import SharedAdam, Worker
import gym
import math, os
# os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt

import pybullet as p
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
from SnakeGymEnv import SnakeGymEnv
import snake
from datetime import datetime

p.connect(p.DIRECT)
# Create env for policy testing.
urdf_path = os.path.join(BASE_DIR, os.pardir, "snake/snake.urdf")
robot = snake.Snake(p, urdf_path)
env = SnakeGymEnv(robot)
# env = gym.make('Pendulum-v0')

if __name__ == '__main__':
	global_network = model.Network(env.observation_space.shape[0], env.action_space.shape[0])
	
	global_network.share_memory()
	optimizer = SharedAdam(global_network.parameters(), lr=0.0002)
	global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

	workers = [Worker(global_network, optimizer, global_ep, global_ep_r, res_queue, i, p, urdf_path) for i in range(16)]
	[w.start() for w in workers]
	res = []
	while True:
		r = res_queue.get()
		if r is not None:
			res.append(r)
		else:
			break
	[w.join() for w in workers]

	plt.plot(res)
	plt.ylabel('Moving average ep reward')
	plt.xlabel('Step')
	plt.show()