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
	obs, r, d, _ = env.step([0.5]*8)
	R += r
	print(d)
	env.render()
	print('Reward: {}'.format(r))
	print("ith step ", i)
print("Total Reward: {}".format(R))
time.sleep(5)