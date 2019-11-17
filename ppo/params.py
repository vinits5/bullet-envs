import argparse
import os
import numpy as np

def params(argv=None):
	parser = argparse.ArgumentParser(description='PPO parameters')
	# Training file parameters
	parser.add_argument('--hidden_size', type=list, default=[256,256], help='Hidden Layers')
	parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate for PPO')
	parser.add_argument('--num_steps', type=int, default=20, help='Steps in each epoch')
	parser.add_argument('--mini_batch_size', type=int, default=5, help='Batch Size for PPO')
	parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of epochs in each PPO update')
	parser.add_argument('--threshold_reward', type=int, default=200, help='Threshold reward to stop training')
	parser.add_argument('--max_frames', type=int, default=15000, help='Maximum frames for training')
	parser.add_argument('--num_envs', type=int, default=16, help='Number of environments for distributed synchronous PPO')
	parser.add_argument('--test_epochs', type=int, default=2, help='Intervals to test the policy')
	parser.add_argument('--resume_training', type=str, default='', help='Resume training')
	parser.add_argument('--log_dir', type=str, default='log', help='Name of the log directory')

	# SnakeGymEnv file parameters
	parser.add_argument('--alpha', type=float, default=1.0, help='Weight applied to forward motion reward')
	parser.add_argument('--beta', type=float, default=0.1, help='Weight applied to drift reward')
	parser.add_argument('--gamma', type=float, default=0.01, help='Weight applied to energy reward')

	# snake file parameters
	parser.add_argument('--selfCollisionEnabled', type=bool, default=True, help='Collision between links')
	parser.add_argument('--motorVelocityLimit', type=float, default=np.inf, help='Joint velocity limit')
	parser.add_argument('--motorTorqueLimit', type=float, default=np.inf, help='Joint torque limit')
	parser.add_argument('--kp', type=float, default=10.0, help='Kp value for joint control')
	parser.add_argument('--kd', type=float, default=0.1, help='Kd value for joint control')
	parser.add_argument('--gaitSelection', type=int, default=1, help='Choose gait')
	parser.add_argument('--scaling_factor', type=float, default=6, help='scaling applied to actions')
	parser.add_argument('--cam_dist', type=float, default=5.0, help='Camera Distance')
	parser.add_argument('--cam_yaw', type=float, default=50, help='Camera Yaw')
	parser.add_argument('--cam_pitch', type=float, default=-35, help='Camera Pitch')
	parser.add_argument('--cam_roll', type=float, default=0, help='Camera Roll')
	parser.add_argument('--upAxisIndex', type=int, default=2, help='Camera Axis')
	parser.add_argument('--render_height', type=int, default=720, help='Height of camera image')
	parser.add_argument('--render_width', type=int, default=960, help='Width of camera image')
	parser.add_argument('--fov', type=int, default=60, help='Field of view for camera')
	parser.add_argument('--nearVal', type=float, default=0.1, help='minimum clipping value for camera')
	parser.add_argument('--farVal', type=float, default=100, help='maximum clipping value for camera')

	args = parser.parse_args(argv)
	return args