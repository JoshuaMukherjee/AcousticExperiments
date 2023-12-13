from acoustools.BEM import BEM_forward_model_grad, grad_H, compute_H
from acoustools.Gorkov import force_mesh, torque_mesh
from acoustools.Mesh import get_weight, get_centre_of_mass_as_points
from acoustools.Utilities import TOP_BOARD, BOTTOM_BOARD

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

def BEM_levitation_objective_top_bottom(transducer_phases, points, board, targets=None, **objective_params):
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

    top_board = TOP_BOARD
    bottom_board = BOTTOM_BOARD

    if "Hgrad" not in objective_params:
        Hx_top, Hy_top, Hz_top = grad_H(None, transducers=top_board, **{"scatterer":scatterer })
        Hx_bottom, Hy_bottom, Hz_bottom = grad_H(None, transducers=bottom_board, **{"scatterer":scatterer })
    else:
        Hx_top, Hy_top, Hz_top, Hx_bottom, Hy_bottom, Hz_bottom = objective_params["Hgrad"]
        
    if "H" not in objective_params:
        H_top = compute_H(scatterer,top_board)
        H_bottom = compute_H(scatterer,bottom_board)
    else:
        H_top,H_bottom = objective_params["H"]

    if "weight" in objective_params:
        weight = objective_params["weight"]
    else:
        weight = -1*get_weight(scatterer)

    force_top = force_mesh(transducer_phases[:,256:,:],points,norms,areas,top_board,None,params,Ax=Hx_top, Ay=Hy_top, Az=Hz_top,F=H_top)
    force_bottom = force_mesh(transducer_phases[:,:256,:],points,norms,areas,bottom_board,None,params,Ax=Hx_bottom, Ay=Hy_bottom, Az=Hz_bottom,F=H_bottom)

    torque_top = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,top_board,force=force_top)
    torque_bottom = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,bottom_board,force=force_bottom)

    torque = torque_top + torque_bottom

    force_x = torch.cat([force_top[:,0,:], force_bottom[:,0,:]])
    force_y = torch.cat([force_top[:,1,:], force_bottom[:,1,:]])
    force_z = torch.cat([force_top[:,2,:], force_bottom[:,2,:]])

    top_board_idx = torch.cat([torch.ones_like(force_top[:,0,:]).to(bool) ,torch.zeros_like(force_bottom[:,0,:]).to(bool)])
    loss_params["top_board_idx"] = top_board_idx

    return torch.sum(loss_function(force_x, force_y, force_z, weight, torque, **loss_params),0).unsqueeze_(0)



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


def balance(force_x, force_y, force_z, weight, torque, **params):

    a,b,c,d,e = params["weights"]

    counter_weight = a*((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    min_torque = b*torch.sum(torque**2,dim=[1,2])
    max_magnitude_x = c*torch.sum((force_x**2))
    max_magnitude_y = d*torch.sum((force_y**2))
    max_magnitude_z =  e*torch.sum((force_z**2))
    return counter_weight + min_torque - max_magnitude_x - max_magnitude_y - max_magnitude_z