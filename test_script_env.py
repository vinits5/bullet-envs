import pybullet as p
import numpy as np
import os
import snake
from SnakeGymEnv import SnakeGymEnv
import time

p.connect(p.GUI)
robot = snake.Snake(p, "snake/snake.urdf")
env = SnakeGymEnv(robot)
obs = env.reset()

print(obs)

R = 0.0
for i in range(60):
	obs, r, _, _ = env.step([1]*8)
	R += r
	print(obs)
	env.render()
	print('Reward: {}'.format(r))

print("Total Reward: {}".format(R))
time.sleep(5)