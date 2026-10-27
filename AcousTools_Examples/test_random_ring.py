from acoustools.Utilities import random_points_ring

import matplotlib.pyplot as plt

pts = random_points_ring(1000, min_r=0.05)


ax = plt.figure().add_subplot(projection='3d')

pts = pts.real

ax.scatter(pts[:,0], pts[:,1], pts[:,2])




pts = random_points_ring(1000, min_r=[0.01, 0.02, 0.05], max_r = [0.02, 0.04, 0.09])

pts = pts.real

ax.scatter(pts[:,0], pts[:,1], pts[:,2])


plt.show()