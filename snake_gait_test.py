import pybullet as p
import numpy as np
import time
import pybullet_data
import matplotlib.pyplot as plt
from PIL import Image

RENDER_HEIGHT = 720
RENDER_WIDTH  = 720

p.connect(p.DIRECT)
p.resetSimulation()

p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("snake/snake.urdf", [0, 0, 0], useFixedBase=0)

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
A = 1
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
			signal.append(A*np.sin(n*s+w*t))
		else:
			signal.append(0)
	return signal

def render():
	base_pos,_ = p.getBasePositionAndOrientation(robot)
	# base_pos = [(0, 0, 0), (0, 0, 0 ,1)]

	view_matrix = p.computeViewMatrixFromYawPitchRoll(
	        						cameraTargetPosition=base_pos,
	        						distance = 1,
							        yaw = 0,
							        pitch = 30,
							        roll = 0,
							        upAxisIndex = 2)
	proj_matrix = p.computeProjectionMatrixFOV(fov = 60,
												aspect = float((RENDER_WIDTH)/RENDER_HEIGHT),
												nearVal = 0.1,
												farVal = 100)

	(_,_,px,_,_) = p.getCameraImage( width = RENDER_WIDTH,
									 height = RENDER_HEIGHT,
									 renderer = p.ER_TINY_RENDERER,
									 viewMatrix = view_matrix,
									 projectionMatrix = proj_matrix)



	rgb_array = np.array(px)
	rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
	# print(rgb_array.shape)
	rgb_array = rgb_array[:,:, :3]

	return rgb_array

def video_logger():
	print("Recording started")
	p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
							"data_video.mp4")
	print("Recording stopped")
# Send commands.
start = time.time()
for i in range(1000):
	t = time.time() - start
	signal = calculateSignal(t)

	p.setJointMotorControlArray(robot, motor_idx.tolist(), p.POSITION_CONTROL, targetPositions=signal)
	p.stepSimulation()
	# img=p.getCameraImage(RENDER_WIDTH,RENDER_HEIGHT, renderer=p.ER_BULLET_HARDWARE_OPENGL)
	if i%100 == 0:
		rgb = render()
		print(rgb.shape)
	# 	video_logger()
	# print(p.getDebugVisualizerCamera(robot))
	# plt.imshow(rgb)
	time.sleep(0.01)

img = Image.fromarray(rgb,'RGB')
img.show()
# plt.show()