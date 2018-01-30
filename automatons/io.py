import numpy as np
import matplotlib.pylab as plt

plt.ion()

data  = np.loadtxt('/home/sheraz/data/cognets/auto/dat.csv', delimiter=',').T

plt.scatter(data[0], data[1])