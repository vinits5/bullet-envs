import pybullet as p
import pybullet_data
import numpy as np
import os
import snake
from SnakeGymEnv import SnakeGymEnv

p.connect(p.DIRECT)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)

robot = snake.Snake(p)

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("snake/snake.urdf", [0, 0, 0], useFixedBase=0)
robot.snake = robot_id

env = SnakeGymEnv(robot)

for i in range(10):
	obs, _,_,_ = env.step([1]*8)
	# print(obs)
while(1):
	print(env.reset(False))