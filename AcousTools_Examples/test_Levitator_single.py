from acoustools.Levitator import LevitatorController

from acoustools.Utilities import BOTTOM_BOARD, create_points, propagate_abs, TOP_BOARD
from acoustools.Solvers import wgs

board = TOP_BOARD

mat_to_world = (1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1)

lev = LevitatorController(ids=(53,), matBoardToWorld=mat_to_world)

p = create_points(1,1,x=0,y=0,z=0)
x = wgs(p, board = board) *0.6
print(propagate_abs(x, p, board = board))



print('starting')
lev.levitate(x)
print('done')

input()
print('stopping')

lev.disconnect()

print('Ended')