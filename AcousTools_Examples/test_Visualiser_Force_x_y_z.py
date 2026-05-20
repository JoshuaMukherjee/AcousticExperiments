if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig, propagate_abs, TRANSDUCERS
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise, ABC
    from acoustools.Force import compute_force, force_fin_diff

    import torch

    p = create_points(1,1,y=0)
    x = wgs(p)
    x = add_lev_sig(x)


    def fx(activations, points):
        force = compute_force(activations, points, TRANSDUCERS)
        f = force.squeeze()[:,0].unsqueeze(0)
        return f
    
    def fy(activations, points):
        force = compute_force(activations, points, TRANSDUCERS)
        f = force.squeeze()[:,1].unsqueeze(0)
        return f
     
    def fz(activations, points):
        force = compute_force(activations, points, TRANSDUCERS)
        f = force.squeeze()[:,2].unsqueeze(0)
        return f

    Visualise(*ABC(0.005, origin=p), x, points=p, 
              colour_functions=[propagate_abs,fx, fy, fz], 
              clr_labels=['Pressure','Analytical Fx','Analytical Fy','Analytical Fz'], res=(200,200),
              link_ax=None, arangement=(2,2))
    