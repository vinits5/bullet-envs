import matplotlib.pyplot as plt
import test
import numpy as np 
import pprint
import time
import snake_gait_test

# log_dir = ''
# max_steps = 30
# create_video = False

# state = test.running_test(log_dir, max_steps=max_steps, create_video=create_video)
# state = np.array(state)

# state_gait = snake_gait_test.test(state.shape[0], create_video=create_video)
# state_gait = np.array(state_gait)
# print(state_gait.shape[0])

state = np.loadtxt('rl_state.txt')
state_gait = np.loadtxt('gait_state.txt')

linewidth = 3
# 1, 7, 11, 15
# Create Plot.
for i in [9]:
	plt.figure(1)
	plt.plot(state[:,i]*(180/np.pi), linewidth=linewidth, label='RL policy')
	plt.plot(state_gait[:,i]*(180/np.pi), linewidth=linewidth, linestyle='--', label='Gait')
	# plt.grid(True)
	plt.legend(fontsize=15, loc=1)
	plt.xlabel('Steps in Simulation', fontsize=25)
	plt.ylabel('Joint Angle (in Degrees)', fontsize=25)
	
	plt.tick_params(labelsize=25, width=3, length=10)
	# plt.figure(2)
	# plt.plot(state[:,i+16])
	# plt.figure(3)
	# plt.plot(state[:,i+16*2])
plt.show()

# save = input('Do you want to save the states [Y/N]: ')
# if save.lower() == 'y':
	# np.savetxt('rl_state.txt', state)
	# np.savetxt('gait_state.txt', state_gait)