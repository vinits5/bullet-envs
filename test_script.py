import pybullet as p
import pybullet_data
import numpy as np
import os
import snake
# from SnakeGymEnv import SnakeGymEnv
gravity = -9.8
# env = SnakeGymEnv()
robot = snake.Snake(p)

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,gravity)
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("snake/snake.urdf", [0, 0, 0], useFixedBase=0)
robot.snake = robot_id

robot.buildMotorList()
# print(robot.motorList)
# print(robot.getVelocity())
# print(robot.getTorque())
# print(robot.getObservation())
while(True):
	robot.step([1]*8)
	print("Working")
