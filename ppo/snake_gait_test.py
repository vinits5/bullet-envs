import pybullet as p
import numpy as np
import time
import pybullet_data

# log videos
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

def log_video(frames):
	out = cv2.VideoWriter('gait_test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
	for frame in frames:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		out.write(frame)
	out.release()

def render():
	cdist = 1.5
	cyaw = -30
	cpitch = -90
	RENDER_WIDTH = 1280
	RENDER_HEIGHT = 720
	p.resetDebugVisualizerCamera(cameraDistance=cdist, cameraYaw=cyaw, cameraPitch=cpitch, cameraTargetPosition=[1.28,0,0])
	(_, _, px, _, _) = p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT)
											   # viewMatrix=view_matrix,
											   # projectionMatrix=proj_matrix)
											   # renderer=self._pybulletClient.ER_BULLET_HARDWARE_OPENGL)
	rgb_array = np.array(px)
	rgb_array = rgb_array[:, :, :3]
	# rgb_array[:,:,0], rgb_array[:,:,2] = rgb_array[:,:,2], rgb_array[:,:,0]
	return rgb_array

def test(max_steps, create_video=False):
	p.connect(p.DIRECT)
	p.resetSimulation()

	p.setAdditionalSearchPath(pybullet_data.getDataPath())

	plane = p.loadURDF("plane.urdf")
	robot = p.loadURDF("../snake/snake.urdf", [0, 0, 0], useFixedBase=0)

	p.setGravity(0, 0, -9.81)   # everything should fall down
	p.setTimeStep(0.01)       # this slows everything down, but let's be accurate...
	p.setRealTimeSimulation(0)  # we want to be faster than real time :)

	# For robot's joint info.
	# for i in range(p.getNumJoints(robot)):
	# 	if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE:
	# 		print('Joint Idx: {}'.format(i))
	# 	if i in [0,1,2,3]:
	# 		print(p.getJointInfo(robot, i)[1])

	motor_idx = np.arange(3,51,3)
	A = np.pi/6
	s = 4
	w = 2

	anistropicFriction = [1, 0.1, 0.01]
	# For base link
	p.changeDynamics(robot, -1, lateralFriction=2, anisotropicFriction=anistropicFriction)
	# For other links
	for i in range(p.getNumJoints(robot)):
		p.getJointInfo(robot, i)
		p.changeDynamics(robot, i, lateralFriction=2, anisotropicFriction=anistropicFriction)

	# print(p.getDynamicsInfo(robot, 0))
	# print(p.getDynamicsInfo(robot, 48))

	# Calculate singal for snake robot.
	def calculateSignal(t):
		signal = []
		for n in range(16):
			if n%2 == 1:
				signal.append(-A*np.sin(n*s+w*t))
			else:
				signal.append(0)
		return signal

	# Send commands.
	start = time.time()
	states = []
	frames = []
	states.append([0]*16)
	print('\nGait Testing Started!')
	for step in range(max_steps):
		t = time.time() - start
		signal = calculateSignal(t)
		states.append(signal)

		p.setJointMotorControlArray(robot, motor_idx.tolist(), p.POSITION_CONTROL, targetPositions=signal)
		p.stepSimulation()
		if create_video: 
			if step%4==0: frames.append(render())

		time.sleep(0.01)
	if create_video: log_video(frames)

	return states