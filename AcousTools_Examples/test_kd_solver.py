from acoustools.Solvers import kd_solver
from acoustools.Utilities import transducers, create_points, propagate, BOARD_POSITIONS
from acoustools.Visualiser import Visualise, ABC

board = transducers(32, BOARD_POSITIONS)
p = create_points(1,1)

x = kd_solver(p,board)
print(x.shape)

print(propagate(x, p, board))
Visualise(*ABC(0.03, origin=p), x, points=p, colour_function_args=[{'board':board}])