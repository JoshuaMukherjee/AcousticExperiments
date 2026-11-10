from acoustools.HighLevel import TopBottomContext
from acoustools.Utilities import create_points
from acoustools.Paths import interpolate_circle
from acoustools.Solvers import kd_solver

p = create_points(1,1,0,0,0)
circle = interpolate_circle(p, 0.02, n=500)


ids = (60,999) 
# ids = (-1,)

with TopBottomContext(default_solver=kd_solver) as ctx:
    ctx.create_focus(circle[0])
    ctx.add_lev_sig()
    ctx.send_solution(ids)
    input()
    while True:
        for c in circle:
            ctx.create_focus(c)
            ctx.add_lev_sig()
            ctx.send_solution(ids)