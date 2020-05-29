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
	#obs:LIDAR data[1,63], x, y
	#action: [right_wheel_ang_vel,left_wheel_ang_vel]
	obs, reward, _, _ = env.step([0, 0])
	print(len(obs))
	count += 1
