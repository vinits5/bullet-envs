import pybullet as p
import numpy as np
import time
import pybullet_data
p.connect(p.GUI)
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

print(p.getDynamicsInfo(robot, 0))
print(p.getDynamicsInfo(robot, 48))

# Calculate singal for snake robot.
def calculateSignal(t):
	signal = []
	for n in range(16):
		if n%2 == 1:
			signal.append(A*np.sin(n*s+w*t))
		else:
			signal.append(0)
	return signal

# Send commands.
start = time.time()
for _ in range(10000):
	t = time.time() - start
	signal = calculateSignal(t)

	p.setJointMotorControlArray(robot, motor_idx.tolist(), p.POSITION_CONTROL, targetPositions=signal)
	p.stepSimulation()

	time.sleep(0.01)