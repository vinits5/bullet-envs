from TurtlebotGymEnv import TurtlebotGymEnv
from turtlebot import Turtlebot
import gym
import pybullet as p

class Args:
	urdf_root = 'turtlebot_urdf/turtlebot.urdf'
	mode = 'train'
	discrete = False
	action_steps = 100
p.connect(p.GUI)
args = Args()
robot = Turtlebot(p, args.urdf_root,args)
env = TurtlebotGymEnv(robot, args)
env.reset()
count = 0

# R = 0.0352
# l = 2*0.176

while count<5000:
	_, done, _, _ = env.step([1, 1])
	# print(done)
	count += 1
	# env.step([0.325, 0.275])
	# x + y = 0.6
	# x - y = 0.05
	# 2x = 0.65
	# x = 0.325
	# y = 0.275

while count<500:
	env.step([1,1])
	count += 1


# 10 rad/s 
