from acoustools.BEM import load_scatterer, get_weight, get_centres_as_points, get_normals_as_points, get_areas, scale_to_radius
import acoustools.Constants as c
from acoustools.Solvers import wgs_wrapper
from acoustools.Utilities import TRANSDUCERS, TOP_BOARD, BOTTOM_BOARD, propagate_abs
from acoustools.Gorkov import force_mesh

from acoustools.Solvers import gradient_descent_solver
from acoustools.Optimise.Constraints import constrain_phase_only

import vedo, torch



def optimise_force_split(transducer_phases, points, board, targets=None, **objective_params):
    '''
    Expects an element of objective_params named `norms` containing all normals in the same order as points and `areas` containing the areas for the cell containing each point. Ignores `board`
    '''

    B = points.shape[0]

    weight = get_weight(scatterer)
    # print("w",weight)
    top_idx = centres[:,2,:] >= 0
    top_idx = torch.unsqueeze(top_idx,1).expand(-1,3,-1)


    top = points[top_idx].reshape(B,3,-1)
    bottom = points[~top_idx].reshape(B,3,-1)


    norms = objective_params["norms"]
    norms_top = norms[top_idx].reshape(B,3,-1)
    norms_bottom = norms[~top_idx].reshape(B,3,-1)

    areas = objective_params["areas"]
    areas = areas.expand(-1,3,-1)

    areas_top = areas[top_idx].reshape(B,3,-1)
    areas_bottom = areas[~top_idx].reshape(B,3,-1)

    transducer_phases_top = transducer_phases[:,:256,:]
    transducer_phases_bottom  = transducer_phases[:,256:,:]


    F_top = force_mesh(transducer_phases_top, top, norms_top,areas_top,board= TOP_BOARD)
    F_bottom = force_mesh(transducer_phases_bottom, bottom, norms_bottom,areas_bottom,board= BOTTOM_BOARD)


    total_force_squared = ((torch.sum(F_top,dim=[1,2])+torch.sum(F_bottom,dim=[1,2])) - weight)**2

    return torch.real(total_force_squared)
    



if __name__ == "__main__":
    path = "Media/Sphere-lam1.stl"
    scatterer = load_scatterer(path,dy=-0.06) #Make mesh at 0,0,0
    
    scale_to_radius(scatterer,0.02)
   
    x1,x2,_,_,_,_ = scatterer.bounds()
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
        "norms":norms,
        "areas":areas
    }
    
    x = gradient_descent_solver(centres, optimise_force_split,constrains=constrain_phase_only,objective_params=params,log=True)
    F = optimise_force_split(x,centres, board, **params)
    print(F)
    print(x)
