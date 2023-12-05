from acoustools.BEM import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas, get_lines_from_plane
from acoustools.Utilities import TRANSDUCERS, propagate_abs
from acoustools.Solvers import gradient_descent_solver
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Visualiser import Visualise

from SymmetricLevitationObjectives import optimise_force_split

import torch


if __name__ == "__main__":
    path = "Media/Sphere-lam1.stl"
    scatterer = load_scatterer(path,dy=-0.06) #Make mesh at 0,0,0
    
    scale_to_diameter(scatterer,0.03)

    print(scatterer.bounds())
    # x1,x2,_,_,_,_ = scatterer.bounds()
    # vedo.show(scatterer)

    board = TRANSDUCERS

    centres = get_centres_as_points(scatterer)

    # x = wgs_wrapper(centres)

    norms = get_normals_as_points(scatterer)

    areas = get_areas(scatterer)

    # centres = centres.expand((2,-1,-1))
    # norms  =norms.expand((2,-1,-1))
    areas = areas.expand((1,-1,-1))
    
    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas
    }
    
    x = gradient_descent_solver(centres, optimise_force_split,constrains=constrain_phase_only,objective_params=params,log=True,iters=400,lr=0.01)
    F = optimise_force_split(x,centres, board, **params)
    print(F)

    A = torch.tensor((-0.07,0, 0.07))
    B = torch.tensor((0.07,0, 0.07))
    C = torch.tensor((-0.07,0, -0.07))

    origin = (0,0,-0.07)
    normal = (0,1,0)

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}

    Visualise(A,B,C,x,colour_functions=[propagate_abs], add_lines_functions=[get_lines_from_plane],add_line_args=[line_params])
    
