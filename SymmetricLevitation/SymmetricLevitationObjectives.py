from acoustools.BEM import load_scatterer, get_weight, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, get_lines_from_plane, get_centre_of_mass_as_points
import acoustools.Constants as c
from acoustools.Solvers import wgs_wrapper
from acoustools.Utilities import TRANSDUCERS, TOP_BOARD, BOTTOM_BOARD, propagate_abs,add_lev_sig
from acoustools.Gorkov import force_mesh, torque_mesh

from acoustools.Solvers import gradient_descent_solver
from acoustools.Optimise.Constraints import constrain_phase_only

from acoustools.Visualiser import Visualise

import vedo, torch



def optimise_force_split(transducer_phases, points, board, targets=None, **objective_params):
    '''
    Expects an element of objective_params named `norms` containing all normals in the same order as points,
    `scatterer` containing the mesh to optimise around and  
    `areas` containing the areas for the cell containing each point. 
    Ignores `board` assuming a top bottom set up
    '''

    B = points.shape[0]

    scatterer = objective_params["scatterer"]
    weight = get_weight(scatterer)
    # print("w",weight)

    top_idx = points[:,2,:] >= 0
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

    # print(torch.sum(F_top,dim=2))
    # print(torch.sum(F_bottom,dim=2))

    total_force_squared = ((torch.sum(F_top,dim=[1,2])-torch.sum(F_bottom,dim=[1,2])) - weight)**2

    return torch.real(total_force_squared)
    
def optimise_force_torque_split(transducer_phases, points, board, targets=None, **objective_params):
    '''
    Expects an element of objective_params named `norms` containing all normals in the same order as points 
    `areas` containing the areas for the cell containing each point
    `scatterer` containing the mesh to optimise around.
    Ignores `board` assuming a top bottom set up
    '''

    B = points.shape[0]

    scatterer = objective_params["scatterer"]
    weight = get_weight(scatterer)
    # print("w",weight)
    top_idx = points[:,2,:] >= 0
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

    centre_of_mass = get_centre_of_mass_as_points(scatterer)
    
    Torque_top = torque_mesh(transducer_phases_top, top, norms_top,areas_top,centre_of_mass=centre_of_mass,board= TOP_BOARD,force=F_top)
    Torque_bottom = torque_mesh(transducer_phases_bottom, bottom, norms_bottom,areas_bottom,centre_of_mass=centre_of_mass,board= BOTTOM_BOARD,force=F_bottom)


    torque = torch.sum(Torque_top,dim=[1,2]) + torch.sum(Torque_bottom,dim=[1,2])
    force = torch.sum(F_top,dim=[1,2])-torch.sum(F_bottom,dim=[1,2])

    # print(torch.sum(F_top,dim=2))
    # print(torch.sum(F_bottom,dim=2))

    total_force_squared = ((torch.sum(F_top,dim=[1,2])-torch.sum(F_bottom,dim=[1,2])) - weight)**2

    return torch.real(total_force_squared)
