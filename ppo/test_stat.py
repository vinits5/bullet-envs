import matplotlib.pyplot as plt
import test
import numpy as np 
import pprint

log_dir = 'log_23_11_2019_20_11_55'
create_video = True

pp = pprint.PrettyPrinter(indent=16)
state = test.running_test(log_dir, create_video=create_video)
state = np.array(state)
# for i in [0,7,15]:
# 	plt.figure(1)
# 	plt.plot(state[:,i])
# 	plt.figure(2)
# 	plt.plot(state[:,i+16])
# 	plt.figure(3)
# 	plt.plot(state[:,i+16*2])
# plt.show()