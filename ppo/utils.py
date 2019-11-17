import matplotlib.pyplot as plt
import torch
import pybullet as p
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
from SnakeGymEnv import SnakeGymEnv
import snake


class IOStream:
	def __init__(self, path, args=None):
		self.f = open(path, 'w')
		if args is not None: self.f.write('%s'%args+'\n')

	def log_params(self, device, num_envs, lr, threshold_reward):
		self.log('Device: {}'.format(device))
		self.log('Number of Environments used: {}'.format(num_envs))
		self.log('Learning Rate: {}'.format(lr))
		self.log('Threshold Reward: {}'.format(threshold_reward))

	def log(self, text):
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()


def logFiles(log_dir):
	code_dir = os.path.join(log_dir, 'code')
	if not os.path.exists(code_dir): os.mkdir(code_dir)
	os.system('cp model.py %s' % (code_dir))
	os.system('cp agent.py %s' % (code_dir))
	os.system('cp utils.py %s'%(code_dir))
	os.system('cp train.py %s'%(code_dir))
	os.system('cp test.py %s'%(code_dir))
	robot_file = os.path.join(os.pardir, 'snake.py')
	os.system('cp %s %s'%(robot_file, code_dir))
	gym_file = os.path.join(os.pardir, 'SnakeGymEnv.py')
	os.system('cp %s %s'%(gym_file, code_dir))

###################### Print Operations #########################
def print_(text="Test", color='w', style='no', bg_color=''):
	color_dict = {'b': 30, 'r': 31, 'g': 32, 'y': 33, 'bl': 34, 'p': 35, 'c': 36, 'w': 37}
	style_dict = {'no': 0, 'bold': 1, 'underline': 2, 'neg1': 3, 'neg2': 5}
	bg_color_dict = {'b': 40, 'r': 41, 'g': 42, 'y': 43, 'bl': 44, 'p': 45, 'c': 46, 'w': 47}
	if bg_color is not '':
		print("\033[" + str(style_dict[style]) + ";" + str(color_dict[color]) + ";" + str(bg_color_dict[bg_color]) + "m" + text + "\033[00m")
	else: print("\033["+ str(style_dict[style]) + ";" + str(color_dict[color]) + "m"+ text + "\033[00m")

def plot(frame_idx, rewards):
	plt.figure(figsize=(20,5))
	plt.subplot(131)
	plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
	plt.plot(rewards)
	plt.show()

def make_env(p, urdf_path, args=None):
	def _thunk():
		robot = snake.Snake(p, urdf_path, args=args)
		env_snake = SnakeGymEnv(robot, args=args)
		return env_snake
	return _thunk

def test_env(env, model, idx, vis=False):
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	state = env.reset()
	if vis: env.render()

	done = False
	total_reward = 0
	steps = 0
	# print('In env')
	print('Test No: {}\r'.format(idx), end="")
	while not done and steps < 10:
		# print('Test in steps')
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		dist, _ = model(state)
		next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0]) #Hack
		state = next_state
		if vis: env.render()
		total_reward += reward
		steps += 1
		# print('End of step')
	return total_reward

