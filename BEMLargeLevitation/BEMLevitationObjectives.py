from acoustools.BEM import BEM_forward_model_grad, grad_H, compute_H
from acoustools.Gorkov import force_mesh, torque_mesh
from acoustools.Mesh import get_weight, get_centre_of_mass_as_points

import torch

def BEM_levitation_objective(transducer_phases, points, board, targets=None, **objective_params):
    ''' Expects an element of objective_params named `norms` containing all normals in the same order as points 
    `areas` containing the areas for the cell containing each point
    `scatterer` containing the mesh to optimise around.'''


    scatterer = objective_params["scatterer"]
    norms = objective_params["norms"]
    areas = objective_params["areas"]

    params = {
        "scatterer":scatterer
    }

    centre_of_mass = get_centre_of_mass_as_points(scatterer)

    Hx, Hy, Hz = grad_H(points=points, transducers=board, **{"scatterer":scatterer })
    H = compute_H(scatterer,board)

    force = force_mesh(transducer_phases,points,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    torque = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,board,force=force)

    if "weight" in objective_params:
        weight = objective_params["weight"]
    else:
        weight = get_weight(scatterer)

    force_x = force[:,0,:]
    force_y = force[:,1,:]
    force_z = force[:,2,:]
    
    return (torch.sum(force_z,dim=1) - weight)**2  + torch.sum(torque,dim=[1,2])**2 + torch.sum(force_x,dim=1)**2 + torch.sum(force_y,dim=1)**2
    