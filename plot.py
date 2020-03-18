import numpy as np
import matplotlib.pyplot as plt

a = np.load("/home/skj/cl/Ddqn0/results/ddqn_SpaceInvadersNoFrameskip-v4_0.npy")
#b = np.load("/home/cl/ddqn-ac/results/ddqn_CartPole-v1_0.npy")

plt.plot(a)
#plt.plot(b)
plt.show()