from TurtlebotGymEnv import TurtlebotGymEnv
import gym
import pybullet as p

class Args:
	urdf_root = 'turtlebot_urdf/turtlebot.urdf'
	mode = 'train'

p.connect(p.GUI)
args = Args()
env = TurtlebotGymEnv(p, args)
env.reset()
count = 0
while True:
	env.step([1,1])
	count += 1