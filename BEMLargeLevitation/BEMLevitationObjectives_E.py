from acoustools.Force import force_mesh, torque_mesh, force_mesh_derivative, get_force_mesh_along_axis
from acoustools.Mesh import get_centre_of_mass_as_points, get_weight
from acoustools.BEM import BEM_forward_model_grad

import torch

def BEM_levitation_objective_subsample_stability_fin_diff_E(transducer_phases, points, board, targets=None, **objective_params):
    ''' Expects an element of objective_params named:\\
    `norms` containing all normals in the same order as points \\
    `areas` containing the areas for the cell containing each point\\
    `scatterer` containing the mesh to optimise around\\
    `indexes` containing subsample to use
    `diff` containing difference to use in fin diffs
    `scatterer_elements` contains list of [object to levitate, walls]\\
    `EGrad` Gradient of E matrix\\
    `E` E matrix
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

    centre_of_mass = get_centre_of_mass_as_points(scatterer)

    Ex, Ey, Ez = objective_params["EGrad"]
    E = objective_params["E"]

    # print("E")
    force = force_mesh(transducer_phases,points,norms,areas,board,Ax=Ex,Ay=Ey,Az=Ez,F=E)
    torque = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,board,force=force)
   
    startX = torch.tensor([[-1*diff],[0],[0]]) 
    endX = torch.tensor([[diff],[0],[0]]) 

    startY = torch.tensor([[0],[-1*diff],[0]])
    endY = torch.tensor([[0],[diff],[0]])

    startZ = torch.tensor([[0],[0],[-1*diff]])
    endZ = torch.tensor([[0],[0],[diff]])
    
    ball = scatter_elems[0]
    walls = scatter_elems[1]

    EsX = objective_params["Ess"][0] 
    ExsX = objective_params["Exss"][0]
    EysX = objective_params["Eyss"][0]
    EzsX = objective_params["Ezss"][0]
    FxsX, _, _ = get_force_mesh_along_axis(startX/1000, endX/1000, transducer_phases, [ball.clone(),walls], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=EsX, Hxs = ExsX, Hys=EysX, Hzs=EzsX)
    
    EsY = objective_params["Ess"][1]
    ExsY = objective_params["Exss"][1]
    EysY = objective_params["Eyss"][1]
    EzsY = objective_params["Ezss"][1]
    _, FysY, _ = get_force_mesh_along_axis(startY/1000, endY/1000, transducer_phases, [ball.clone(),walls], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=EsY, Hxs = ExsY, Hys=EysY, Hzs=EzsY)
    
    EsZ = objective_params["Ess"][2]
    ExsZ = objective_params["Exss"][2]
    EysZ = objective_params["Eyss"][2]
    EzsZ = objective_params["Ezss"][2]
    _, _, FzsZ = get_force_mesh_along_axis(startZ/1000, endZ/1000, transducer_phases, [ball.clone(),walls], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=EsZ, Hxs = ExsZ, Hys=EysZ, Hzs=EzsZ)



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


def levitation_balance_greater_grad_torque(force_x, force_y, force_z, weight, torque, **params):
    a,b,c,d = params["weights"]

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    net_x = torch.sum(force_x)**2
    net_y = torch.sum(force_y)**2
    net_z = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    # counter_weight = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    
    balance = a* (net_x + net_y + net_z)

    greater_weight = b*(weight - torch.sum(force_z) ).unsqueeze_(0)


    grad_X = FxsX[0] - FxsX[-1]
    grad_Y = FysY[0] - FysY[-1]
    grad_Z = FzsZ[0] - FzsZ[-1]
    gradient = -1 *c * (grad_X + grad_Y + grad_Z)

    min_torque = d*torch.sum(torque**2,dim=[1,2])

    return balance + greater_weight + gradient + min_torque

def levitation_balance_mag_grad_torque(force_x, force_y, force_z, weight, torque, **params):
    a,b,c,d = params["weights"]
    # print(force_x)

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    net_x = torch.sum(force_x)**2
    net_y = torch.sum(force_y)**2
    net_z = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    # counter_weight = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    
    balance = a* (net_x + net_y + net_z)

    mag_x = torch.sum(force_x**2)
    mag_y = torch.sum(force_y**2)
    mag_z = torch.sum(force_z**2)
    magnitude = -1 * b * (mag_x + mag_y + mag_z)

    grad_X = FxsX[0] - FxsX[-1]
    grad_Y = FysY[0] - FysY[-1]
    grad_Z = FzsZ[0] - FzsZ[-1]
    gradient = -1 *c * (grad_X + grad_Y + grad_Z)

    min_torque = d*torch.sum(torque**2,dim=[1,2])

    # print(balance, magnitude, gradient, min_torque)
    return balance + magnitude + gradient + min_torque