from acoustools.Paths import interpolate_points
from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Solvers import naive
from acoustools.Export.CSV import write_to_file


start = create_points(1,1,0,0,0)
end = create_points(1,1,0,0,-0.03)

path = interpolate_points(start, end, 100)

board = TRANSDUCERS

xs = []

for p in path:
    x = naive(p, board)

    xs.append(x)

write_to_file(xs, './AcousTools_Examples/data/test_line_down.csv', len(xs))