if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig, propagate_abs
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise, ABC
    from acoustools.Force import compute_force, force_fin_diff

    import torch

    p = create_points(1,1,y=0)
    x = wgs(p)
    x = add_lev_sig(x)


    Visualise(*ABC(0.01, origin=p), x, points=p, 
              colour_functions=[propagate_abs,compute_force], 
              clr_labels=['Pressure','Analytical Force'], res=(200,200),link_ax=None)