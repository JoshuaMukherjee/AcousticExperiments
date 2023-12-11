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

    loss_function = objective_params["loss"]
    
    if "loss_params" in objective_params:
        loss_params = objective_params["loss_params"]
    else:
        loss_params = {} 

    params = {
        "scatterer":scatterer
    }

    centre_of_mass = get_centre_of_mass_as_points(scatterer)

    if "Hgrad" not in objective_params:
        Hx, Hy, Hz = grad_H(None, transducers=board, **{"scatterer":scatterer })
    else:
        Hx, Hy, Hz = objective_params["Hgrad"]
    
    if "H" not in objective_params:
        H = compute_H(scatterer,board)
    else:
        H = objective_params["H"]

    force = force_mesh(transducer_phases,points,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    torque = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,board,force=force)

    if "weight" in objective_params:
        weight = objective_params["weight"]
    else:
        weight = -1*get_weight(scatterer)

    force_x = force[:,0,:]
    force_y = force[:,1,:]
    force_z = force[:,2,:]
    
    return loss_function(force_x, force_y, force_z, weight, torque, **loss_params)

def sum_forces_torque(force_x, force_y, force_z, weight, torque, **params):

    return (torch.sum(force_z,dim=1) - weight)**2  + torch.sum(torque,dim=[1,2])**2 + torch.sum(force_x,dim=1)**2 + torch.sum(force_y,dim=1)**2


def sum_top_bottom_force_torque(force_x, force_y, force_z, weight, torque, **params):
    top_board_idx = params["top_board_idx"]
    
    force_z_top =force_z[top_board_idx]
    force_z_bottom = force_z[~top_board_idx]

    # print("Z-W",torch.sum((force_z_bottom - weight)**2).unsqueeze_(0))
    # print("T", torch.sum(torque**2,dim=[1,2]))
    # print("X", torch.sum(force_x**2,dim=1))
    # print("Y", torch.sum(force_y**2,dim=1))
    # print("Z", torch.sum(force_z_top**2))

    # print(torch.sum((force_z_bottom)), weight,torch.sum(force_z_top) )

    return (torch.sum((force_z_bottom)) + (weight+torch.sum(force_z_top)) )**2 + torch.sum(torque**2,dim=[1,2]) + torch.sum(force_x**2,dim=1) + torch.sum(force_y**2,dim=1)  - 1e-2*(torch.sum((force_z_bottom))**2)

def max_magnitude_min_force(force_x, force_y, force_z, weight, torque, **params):
    '''
    Minimise the net force and torque while maximising the magnitudes of these forces \\
    Needs a parameter for the top and bottom boards contributions eg `indexes = (centres[:,2,:] > centre_of_mass[:,2,:])` for a sphere or similar object
    '''
    top_board_idx = params["top_board_idx"]
    a,b,c,d,e,f,g = params["weights"]
    
    force_z_top =force_z[top_board_idx]
    force_z_bottom = force_z[~top_board_idx]

    counter_weight = (torch.sum((force_z_bottom)) + (weight+torch.sum(force_z_top)) )**2
    
    min_torque = torch.sum(torque**2,dim=[1,2])
    min_x = torch.sum(force_x,dim=1)**2
    min_y = torch.sum(force_y,dim=1)**2
    
    max_magnitude_z = torch.sum((force_z_bottom))**2
    max_magnitude_x = torch.sum((force_x**2))
    max_magnitude_y = torch.sum((force_y**2))

    # print(a*counter_weight , b*min_torque , c*min_x , d*min_y  , e*max_magnitude_z  , f*max_magnitude_x , g*max_magnitude_y, sep="\n")

    return a*counter_weight + b*min_torque + c*min_x + d*min_y  - e*max_magnitude_z  - f*max_magnitude_x - g*max_magnitude_y


