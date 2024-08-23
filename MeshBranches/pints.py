import numpy as np
import trimesh


points = np.random.random((100,3))

points[np.random.random_integers(0,99,(1,))] += [0, 10, 0]
points[np.random.random_integers(0,99,(1,))] += [0, -10, 0]
points[np.random.random_integers(0,99,(1,))] += [0, -0, 10]

mesh = trimesh.Trimesh(points, process=False).convex_hull


mesh.show()