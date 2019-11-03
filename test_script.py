import pybullet as p
import pybullet_data
import numpy as np
import os
import snake
from SnakeGymEnv import SnakeGymEnv

# env = SnakeGymEnv()
robot = snake.Snake(p)

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("snake/snake.urdf", [0, 0, 0], useFixedBase=0)
robot.snake = robot_id


print(robot.getBasePosition())


