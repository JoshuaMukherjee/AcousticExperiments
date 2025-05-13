from acoustools.Solvers import kd_solver
from acoustools.Utilities import transducers, create_points, propagate
from acoustools.Visualiser import Visualise, ABC

board = transducers(4, 0.04)
p = create_points(1,1,0,0,0)

x = kd_solver(p,board)
print(x.shape)

print(propagate(x, p, board))
Visualise(*ABC(0.03), x, points=p, colour_function_args=[{'board':board}])