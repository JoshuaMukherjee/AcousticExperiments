from acoustools.BEM import BEM_forward_model_grad, grad_H, compute_H, grad_2_H, get_cache_or_compute_H_gradients, get_cache_or_compute_H
from acoustools.Gorkov import force_mesh, torque_mesh, force_mesh_derivative, get_force_mesh_along_axis
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

def BEM_levitation_objective_subsample(transducer_phases, points, board, targets=None, **objective_params):
    ''' Expects an element of objective_params named:\\
    `norms` containing all normals in the same order as points \\
    `areas` containing the areas for the cell containing each point\\
    `scatterer` containing the mesh to optimise around\\
    `indexes` containing subsample to use'''


    scatterer = objective_params["scatterer"]
    norms = objective_params["norms"]
    areas = objective_params["areas"]
    indexes = objective_params["indexes"]

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

    force_x = force[:,0,:][:,indexes]
    force_y = force[:,1,:][:,indexes]
    force_z = force[:,2,:][:,indexes]
    torque = torque[:,:,indexes]
    
    return loss_function(force_x, force_y, force_z, weight, torque, **loss_params)

def BEM_levitation_objective_subsample_stability(transducer_phases, points, board, targets=None, **objective_params):
    ''' Expects an element of objective_params named:\\
    `norms` containing all normals in the same order as points \\
    `areas` containing the areas for the cell containing each point\\
    `scatterer` containing the mesh to optimise around\\
    `indexes` containing subsample to use
    Also uses the derivative of the force to add a stability component
    '''


    scatterer = objective_params["scatterer"]
    norms = objective_params["norms"]
    areas = objective_params["areas"]
    indexes = objective_params["indexes"]

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

    if "Hgrad2" not in objective_params:
        Haa = grad_2_H(None, transducers=board, **{"scatterer":scatterer })
    else:
        Haa = objective_params["Hgrad2"]



    force = force_mesh(transducer_phases,points,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    torque = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,board,force=force)
    force_grad = force_mesh_derivative(transducer_phases,points, norms, areas, board, scatterer, Hx=Hx, Hy=Hy, Hz=Hz,Haa=Haa)

    if "weight" in objective_params:
        weight = objective_params["weight"]
    else:
        weight = -1*get_weight(scatterer)

    force_x = force[:,0,:][:,indexes]
    force_y = force[:,1,:][:,indexes]
    force_z = force[:,2,:][:,indexes]
    force_grad = force_grad[:,:,indexes]
    torque = torque[:,:,indexes]

    loss_params["force_grad"] = force_grad
    
    return loss_function(force_x, force_y, force_z, weight, torque, **loss_params)

def BEM_levitation_objective_subsample_stability_fin_diff(transducer_phases, points, board, targets=None, **objective_params):
    ''' Expects an element of objective_params named:\\
    `norms` containing all normals in the same order as points \\
    `areas` containing the areas for the cell containing each point\\
    `scatterer` containing the mesh to optimise around\\
    `indexes` containing subsample to use
    `diff` containing difference to use in fin diffs
    `scatterer_elements` contains list of [object to levitate, walls]
    '''


    scatterer = objective_params["scatterer"]
    norms = objective_params["norms"]
    areas = objective_params["areas"]
    indexes = objective_params["indexes"]
    diff = objective_params["diff"]
    scatter_elems = objective_params["scatterer_elements"]

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
        Hx, Hy, Hz = get_cache_or_compute_H_gradients(None, transducers=board, **{"scatterer":scatterer })
    else:
        Hx, Hy, Hz = objective_params["Hgrad"]
    
    if "H" not in objective_params:
        H = get_cache_or_compute_H(scatterer,board)
    else:
        H = objective_params["H"]


    force = force_mesh(transducer_phases,points,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    torque = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,board,force=force)
   
    
    startX = torch.tensor([[-1*diff],[0],[0]])
    endX = torch.tensor([[diff],[0],[0]])

    startY = torch.tensor([[0],[-1*diff],[0]])
    endY = torch.tensor([[0],[diff],[0]])

    startZ = torch.tensor([[0],[0],[-1*diff]])
    endZ = torch.tensor([[0],[0],[diff]])
    
    ball = scatter_elems[0]
    walls = scatter_elems[1]
    FxsX, _, _ = get_force_mesh_along_axis(startX, endX, transducer_phases, [ball.clone(),walls], board,indexes,steps=3, use_cache=True, print_lines=False)
    _, FysY, _ = get_force_mesh_along_axis(startY, endY, transducer_phases, [ball.clone(),walls], board,indexes,steps=3, use_cache=True, print_lines=False)
    _, _, FzsZ = get_force_mesh_along_axis(startZ, endZ, transducer_phases, [ball.clone(),walls], board,indexes,steps=3, use_cache=True, print_lines=False)



    if "weight" in objective_params:
        weight = objective_params["weight"]
    else:
        weight = -1*get_weight(scatterer)

    force_x = force[:,0,:][:,indexes]
    force_y = force[:,1,:][:,indexes]
    force_z = force[:,2,:][:,indexes]
    torque = torque[:,:,indexes]

    loss_params["FxsX"]=FxsX
    loss_params["FysY"]=FysY
    loss_params["FzsZ"]=FzsZ

    
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

def balance(force_x, force_y, force_z, weight, torque, **params):

    a,b,c,d,e = params["weights"]

    counter_weight = a*((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    min_torque = b*torch.sum(torque**2,dim=[1,2])
    max_magnitude_x = c*torch.sum((force_x**2))
    max_magnitude_y = d*torch.sum((force_y**2))
    max_magnitude_z =  e*torch.sum(torch.abs(force_z))
    return counter_weight + min_torque - max_magnitude_x - max_magnitude_y - max_magnitude_z

def balance_max_z(force_x, force_y, force_z, weight, torque, **params):

    a,b,c,d,e = params["weights"]

    counter_weight = a*((weight - torch.sum(force_z))).unsqueeze_(0) #different to `balance` on this line
    min_torque = b*torch.sum(torch.abs(torque),dim=[1,2])
    max_magnitude_x = c*torch.sum(torch.abs(force_x))
    max_magnitude_y = d*torch.sum(torch.abs(force_y))
    max_magnitude_z =  e*torch.sum(torch.abs(force_z))
    return counter_weight + min_torque - max_magnitude_x - max_magnitude_y - max_magnitude_z

def weight_force(force_x, force_y, force_z, weight, torque, **params):
    a,b,c,d,e = params["weights"]

    counter_weight = abs(weight) - torch.sum(force_z)
    # max_z = -1 * torch.sum(force_z)
    # print(counter_weight , max_z)
    return (a*counter_weight ).unsqueeze_(0)

def balance_greater_z(force_x, force_y, force_z, weight, torque, **params):
    '''
    Aims for force as close to zero with Fz < mg
    '''

    a,b,c,d,e,f = params["weights"]

    counter_weight = a*((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    min_torque = b*torch.sum(torque**2,dim=[1,2])
    
    max_magnitude_x = c*torch.sum((force_x**2))
    max_magnitude_y = d*torch.sum((force_y**2))
    max_magnitude_z =  e*torch.sum(torch.abs(force_z))
    
    f_z_greater = f*((weight - torch.sum(force_z))).unsqueeze_(0) #different to `balance` on this line

    return counter_weight + min_torque - max_magnitude_x - max_magnitude_y - max_magnitude_z + f_z_greater

def balance_stability(force_x, force_y, force_z, weight, torque, **params):
    '''
    Balance with gradient of force as a stability measure 
    '''

    a,b,c,d,e,f = params["weights"]

    counter_weight = a*((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    min_torque = b*torch.sum(torque**2,dim=[1,2])
    max_magnitude_x = c*torch.sum((force_x**2))
    max_magnitude_y = d*torch.sum((force_y**2))
    max_magnitude_z =  e*torch.sum(torch.abs(force_z))

    force_grad = params["force_grad"]
    stability_loss = f*torch.sum(force_grad.real)

    return counter_weight + min_torque - max_magnitude_x - max_magnitude_y - max_magnitude_z + stability_loss

def balance_greater_z_stability(force_x, force_y, force_z, weight, torque, **params):
    '''
    Aims for force as close to zero with Fz < mg
    '''

    a,b,c,d,e,f,g = params["weights"]

    counter_weight = a*((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    min_torque = b*torch.sum(torque**2,dim=[1,2])
    
    max_magnitude_x = c*torch.sum((force_x**2))
    max_magnitude_y = d*torch.sum((force_y**2))
    max_magnitude_z =  e*torch.sum(torch.abs(force_z))
    
    f_z_greater = f*((weight - torch.sum(force_z))).unsqueeze_(0) #different to `balance` on this line

    force_grad = params["force_grad"]
    stability_loss = g*torch.sum(force_grad.real)

    return counter_weight + min_torque - max_magnitude_x - max_magnitude_y - max_magnitude_z + f_z_greater + stability_loss

def balance_greater_z_stability_equal(force_x, force_y, force_z, weight, torque, **params):
    '''
    Aims for force as close to zero with Fz < mg
    '''

    a,b,c,d,e,f,g,h,i = params["weights"]

    counter_weight = a*((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    min_torque = b*torch.sum(torque**2,dim=[1,2])
    
    max_magnitude_x = c*torch.sum((force_x**2))
    max_magnitude_y = d*torch.sum((force_y**2))
    max_magnitude_z =  e*torch.sum(torch.abs(force_z))
    
    f_z_greater = f*((weight - torch.sum(force_z))).unsqueeze_(0) #different to `balance` on this line

    force_grad = params["force_grad"]
    stability_loss = g*torch.sum(force_grad.real)

    net_x = h*torch.sum(force_x)**2
    net_y = i*torch.sum(force_y)**2

    return counter_weight + min_torque - max_magnitude_x - max_magnitude_y - max_magnitude_z + f_z_greater + stability_loss + net_x + net_y


def balance_greater_z_stab_fin_diff(force_x, force_y, force_z, weight, torque, **params):

    a,b,c,d,e,f,g,h,i = params["weights"]

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    counter_weight = a*((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    min_torque = b*torch.sum(torque**2,dim=[1,2])

    max_magnitude_x = c*torch.sum((force_x**2))
    max_magnitude_y = d*torch.sum((force_y**2))
    max_magnitude_z =  e*torch.sum(torch.abs(force_z))

    f_z_greater = f*((weight - torch.sum(force_z))).unsqueeze_(0) #different to `balance` on this line

    stab_X = g*(FxsX[0] - FxsX[2])
    stab_Y = h*(FysY[0] - FysY[2])
    stab_Z = i*(FzsZ[0] - FzsZ[2])

    return counter_weight + min_torque - max_magnitude_x - max_magnitude_y - max_magnitude_z + f_z_greater - stab_X - stab_Y - stab_Z

