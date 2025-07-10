if __name__ == "__main__":
    from acoustools.Solvers import gradient_descent_solver, gorkov_target
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, generate_pressure_targets, generate_gorkov_targets, TRANSDUCERS
    from acoustools.Optimise.Objectives import propagate_abs_sum_objective, gorkov_analytical_sum_objective, pressure_abs_gorkov_trapping_stiffness_objective, target_pressure_mse_objective, target_gorkov_mse_objective
    from acoustools.Optimise.Constraints import *
    from acoustools.Gorkov import gorkov_analytical
    from acoustools.Visualiser import Visualise, ABC

    import torch    

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy 

    board = TRANSDUCERS

    p = create_points(1,1,0,0,0)
    targets_u = generate_gorkov_targets(1,1,min_val=-8,max_val=-6)
    x5 = gradient_descent_solver(p,target_gorkov_mse_objective, board=board,
                                    constrains=sine_amplitude, lr=1e8, iters=500, targets=targets_u,log=True)
    


    # x5 = add_lev_sig(x5)

    print(torch.abs(x5))
    
    print(targets_u, gorkov_analytical(x5, p, board))

    Visualise(*ABC(0.05, origin=p), x5, colour_functions=[propagate_abs, gorkov_analytical], res=(150,150), link_ax=None)