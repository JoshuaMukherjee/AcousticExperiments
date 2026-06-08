if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig, propagate_abs
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise, ABC
    from acoustools.Gorkov import gorkov_analytical, gorkov_fin_diff, gorkov_autograd

    import torch

    p = create_points(1,1,0,0,0)
    x = wgs(p)
    x = add_lev_sig(x)


    Visualise(*ABC(0.01, origin=p), x, points=p, 
              colour_functions=[propagate_abs,gorkov_analytical, gorkov_fin_diff, gorkov_autograd], 
              clr_labels=['Pressure','Analytical Gor\'kov', 'Finite Differences Gor\'kov', 'Autodiff Gor\'kov'], res=(200,200),link_ax=[1,2,3])