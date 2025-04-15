from acoustools.Utilities import create_points, propagate_velocity_real, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, ABC

p = create_points(5,1)

x = wgs(p)


# Visualise(*ABC(0.05), x, p, colour_functions=[propagate_abs])
Visualise(*ABC(0.05, plane='xz'), x, colour_functions=[propagate_abs,propagate_velocity_real], link_ax=None, call_abs=True)