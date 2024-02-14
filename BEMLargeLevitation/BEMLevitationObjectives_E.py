from acoustools.Force import force_mesh, torque_mesh, force_mesh_derivative, get_force_mesh_along_axis
from acoustools.Mesh import get_centre_of_mass_as_points, get_weight
from acoustools.BEM import BEM_forward_model_grad
import acoustools.Constants as Constants

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

def balance_sign_mag(force_x, force_y, force_z, weight, torque, **params):
    a,b,c = params["weights"]
    
    norms = params["norms"].real
    norms_x = norms[:,0,:]
    norms_y = norms[:,1,:]
    norms_z = norms[:,2,:]

    net_x = torch.sum(force_x)**2
    net_y = torch.sum(force_y)**2
    net_z = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    balance = a* (net_x + net_y + net_z)

    mag_x = torch.sum(torch.abs(force_x))
    mag_y = torch.sum(torch.abs(force_y))
    mag_z = torch.sum(torch.abs(force_z))
    magnitude = -1*b * (mag_x + mag_y + mag_z)

    force = torch.stack([force_x, force_y, force_z],axis=1) #Book 1 Pg. 280
    alphas = force/norms
    sign = c*torch.sum(torch.maximum(torch.zeros_like(alphas),alphas))**2

    # sign_x = torch.sum(torch.abs(torch.sign(force_x) + torch.sign(norms_x)))
    # sign_y = torch.sum(torch.abs(torch.sign(force_y) + torch.sign(norms_y)))
    # sign_z = torch.sum(torch.abs(torch.sign(force_z) + torch.sign(norms_z)))
    # sign = c/6 * (sign_x + sign_y + sign_z) # 1/6 normalises as 3 axis and when signs equal loss = 2 see Book 1 Pg. 280
    # # print(sign)
    # force = torch.stack([force_x, force_y, force_z],axis=1)
    # # print(torch.sum(norms*force,1)/(torch.norm(force,2)*torch.norm(norms,2)))
    # print(norms*force)
    # print(torch.sum(force,dim=1))
    # print(torch.norm(force,2,1))
    # print(norms)
    # angle = (torch.sum(norms*force,1)/torch.norm(force,2,1))
    # print(angle)
    # sign = torch.sum((torch.pi - angle)**2)
    # exit()
    # print(sign)
    print(balance, magnitude, sign)
    return balance + magnitude + sign

def BEM_E_pressure_objective(transducer_phases, points, board, targets=None, **objective_params):

    loss_function = objective_params["loss"]
    E = objective_params["E"]
    Ex,Ey,Ez = objective_params["EGrad"]

    scatterer = objective_params["scatterer"]
    norms = objective_params["norms"]
    areas = objective_params["areas"]
    indexes = objective_params["indexes"]
    weight = objective_params["weight"]
    loss_params = objective_params["loss_params"]

    return loss_function(transducer_phases,points,norms,areas,board,weight,indexes,Ax=Ex,Ay=Ey,Az=Ez,F=E,**loss_params)


def pressure_direction_loss(transducer_phases,points,normals,areas,board,weight,indexes,Ax,Ay,Az,F, **params):

    a,b,c,d = params["weights"]

    pressure = torch.abs(F@transducer_phases)[:,indexes]
    pressure_grad_x = torch.abs(Ax@transducer_phases)[:,indexes]
    pressure_grad_y = torch.abs(Ay@transducer_phases)[:,indexes]
    pressure_grad_z = torch.abs(Az@transducer_phases)[:,indexes]
    pressure_grad = torch.stack([pressure_grad_x,pressure_grad_y,pressure_grad_z],dim=1)

    force = force_mesh(transducer_phases,points,normals,areas,board=board,F=F,Ax=Ax,Ay=Ay,Az=Az)[:,:,indexes]
    force[:,2,:] += weight

    alpha = torch.abs(pressure)**2 - 1/Constants.k**2 * torch.norm(pressure_grad,p=2,dim=1)**2
    alpha = alpha.permute(0,2,1)
    # f = 1/(4 * Constants.p_0 * Constants.c_0**2) * areas[:,:,indexes] * alpha * normals[:,:,indexes]
    # print(force, f, alpha)
    alpha = torch.sum(alpha)
    # pen = torch.sum(torch.maximum(torch.zeros_like(alpha),alpha))

    max_p = d* torch.sum(torch.abs(pressure))

    loss = a*torch.sum(force)**2 - b*torch.sum(force**2) + c*alpha +max_p
    # print(a*torch.sum(force)**2 , b*torch.sum(force**2) , c*alpha)
    return loss.unsqueeze(0)