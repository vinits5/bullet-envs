import matplotlib.pyplot as plt
import test
import numpy as np 
import pprint

pp = pprint.PrettyPrinter(indent=16)
state = test.running_test()
state = np.array(state)
for i in [0,7,15]:
	plt.figure(1)
	plt.plot(state[:,i])
	plt.figure(2)
	plt.plot(state[:,i+16])
	plt.figure(3)
	plt.plot(state[:,i+16*2])
plt.show()