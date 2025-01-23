from acoustools.BEM import BEM_forward_model_grad, grad_H, compute_H, grad_2_H, get_cache_or_compute_H_gradients, get_cache_or_compute_H, compute_E
from acoustools.Force import force_mesh, torque_mesh, force_mesh_derivative, get_force_mesh_along_axis
from acoustools.Mesh import get_weight, get_centre_of_mass_as_points
from acoustools.Utilities import TOP_BOARD, BOTTOM_BOARD

from BEMLevUtils import get_H_for_fin_diffs

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

    # E = compute_E(scatterer,points,board, H=H)
    force = force_mesh(transducer_phases,points,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    # force = force_mesh(transducer_phases,points,norms,areas,board,BEM_forward_model_grad,params,F=E)
    torque = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,board,force=force)
   
    startX = torch.tensor([[-1*diff],[0],[0]]) 
    endX = torch.tensor([[diff],[0],[0]]) 

    startY = torch.tensor([[0],[-1*diff],[0]])
    endY = torch.tensor([[0],[diff],[0]])

    startZ = torch.tensor([[0],[0],[-1*diff]])
    endZ = torch.tensor([[0],[0],[diff]])
    
    ball = scatter_elems[0]
    walls = scatter_elems[1]

    if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startX/1000, endX/1000, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
    else:
        Hs = objective_params["Hss"][0]
        Hxs = objective_params["Hxss"][0]
        Hys = objective_params["Hyss"][0]
        Hzs = objective_params["Hzss"][0]

    FxsX, _, _ = get_force_mesh_along_axis(startX/1000, endX/1000, transducer_phases, [ball.clone(),walls], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs)
    
    if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startY/1000, endY/1000, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
    else:
        Hs = objective_params["Hss"][1]
        Hxs = objective_params["Hxss"][1]
        Hys = objective_params["Hyss"][1]
        Hzs = objective_params["Hzss"][1]

    _, FysY, _ = get_force_mesh_along_axis(startY/1000, endY/1000, transducer_phases, [ball.clone(),walls], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs)
    
    if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startZ/1000, endZ/1000, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
    else:
        Hs = objective_params["Hss"][2]
        Hxs = objective_params["Hxss"][2]
        Hys = objective_params["Hyss"][2]
        Hzs = objective_params["Hzss"][2]

    _, _, FzsZ = get_force_mesh_along_axis(startZ/1000, endZ/1000, transducer_phases, [ball.clone(),walls], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs)



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

def BEM_levitation_objective_subsample_stability_fin_diff_single_element(transducer_phases, points, board, targets=None, **objective_params):
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

    # E = compute_E(scatterer,points,board, H=H)
    force = force_mesh(transducer_phases,points,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    # force = force_mesh(transducer_phases,points,norms,areas,board,BEM_forward_model_grad,params,F=E)
    torque = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,board,force=force)
   
    startX = torch.tensor([[-1*diff],[0],[0]]) 
    endX = torch.tensor([[diff],[0],[0]]) 

    startY = torch.tensor([[0],[-1*diff],[0]])
    endY = torch.tensor([[0],[diff],[0]])

    startZ = torch.tensor([[0],[0],[-1*diff]])
    endZ = torch.tensor([[0],[0],[diff]])
    
    ball = scatter_elems[0]

    if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startX/1000, endX/1000, [ball.clone()], board, steps=1, use_cache=True, print_lines=False)
    else:
        Hs = objective_params["Hss"][0]
        Hxs = objective_params["Hxss"][0]
        Hys = objective_params["Hyss"][0]
        Hzs = objective_params["Hzss"][0]

    FxsX, _, _ = get_force_mesh_along_axis(startX/1000, endX/1000, transducer_phases, [ball.clone()], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs)
    
    if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startY/1000, endY/1000, [ball.clone()], board, steps=1, use_cache=True, print_lines=False)
    else:
        Hs = objective_params["Hss"][1]
        Hxs = objective_params["Hxss"][1]
        Hys = objective_params["Hyss"][1]
        Hzs = objective_params["Hzss"][1]

    _, FysY, _ = get_force_mesh_along_axis(startY/1000, endY/1000, transducer_phases, [ball.clone()], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs)
    
    if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startZ/1000, endZ/1000, [ball.clone()], board, steps=1, use_cache=True, print_lines=False)
    else:
        Hs = objective_params["Hss"][2]
        Hxs = objective_params["Hxss"][2]
        Hys = objective_params["Hyss"][2]
        Hzs = objective_params["Hzss"][2]

    _, _, FzsZ = get_force_mesh_along_axis(startZ/1000, endZ/1000, transducer_phases, [ball.clone()], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs)



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


def BEM_levitation_objective_subsample_stability_fin_diff_jacobian(transducer_phases, points, board, targets=None, **objective_params):
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

    # E = compute_E(scatterer,points,board, H=H)
    force = force_mesh(transducer_phases,points,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    # force = force_mesh(transducer_phases,points,norms,areas,board,BEM_forward_model_grad,params,F=E)
    torque = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,board,force=force)
   
    startX = torch.tensor([[-1*diff],[0],[0]]) 
    endX = torch.tensor([[diff],[0],[0]]) 

    startY = torch.tensor([[0],[-1*diff],[0]])
    endY = torch.tensor([[0],[diff],[0]])

    startZ = torch.tensor([[0],[0],[-1*diff]])
    endZ = torch.tensor([[0],[0],[diff]])
    
    ball = scatter_elems[0]
    walls = scatter_elems[1]

    if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startX/1000, endX/1000, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
    else:
        Hs = objective_params["Hss"][0]
        Hxs = objective_params["Hxss"][0]
        Hys = objective_params["Hyss"][0]
        Hzs = objective_params["Hzss"][0]

    FxsX, FysX, FzsX = get_force_mesh_along_axis(startX/1000, endX/1000, transducer_phases, [ball.clone(),walls], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs)
    
    if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startY/1000, endY/1000, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
    else:
        Hs = objective_params["Hss"][1]
        Hxs = objective_params["Hxss"][1]
        Hys = objective_params["Hyss"][1]
        Hzs = objective_params["Hzss"][1]

    FxsY, FysY, FzsY = get_force_mesh_along_axis(startY/1000, endY/1000, transducer_phases, [ball.clone(),walls], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs)
    
    if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startZ/1000, endZ/1000, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
    else:
        Hs = objective_params["Hss"][2]
        Hxs = objective_params["Hxss"][2]
        Hys = objective_params["Hyss"][2]
        Hzs = objective_params["Hzss"][2]

    FxsZ, FysZ, FzsZ = get_force_mesh_along_axis(startZ/1000, endZ/1000, transducer_phases, [ball.clone(),walls], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs)

    print(FxsZ)

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

    a,b,c,d,e,f,g,h,i,j = params["weights"]

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    counter_weight = a*((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    min_torque = b*torch.sum(torque**2,dim=[1,2])

    max_magnitude_x = c*torch.sum((force_x**2))
    max_magnitude_y = d*torch.sum((force_y**2))

    f_z_greater = e*((weight - torch.sum(force_z))).unsqueeze_(0) #different to `balance` on this line

    stab_X = f*(FxsX[0] - FxsX[-1])
    stab_Y = g*(FysY[0] - FysY[-1])
    stab_Z = h*(FzsZ[0] - FzsZ[-1])

    net_x = i*torch.sum(force_x)**2
    net_y = j*torch.sum(force_y)**2

    return counter_weight + min_torque - max_magnitude_x - max_magnitude_y + f_z_greater - stab_X - stab_Y - stab_Z + net_x + net_y


def levitation_balance_magnitude_grad_fin_diff(force_x, force_y, force_z, weight, torque, **params):
    a,b,c = params["weights"]

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    net_x = torch.sum(force_x)**2
    net_y = torch.sum(force_y)**2
    counter_weight = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    balance = a* (net_x + net_y + counter_weight)

    mag_x = torch.sum(force_x**2)
    mag_y = torch.sum(force_y**2)
    mag_z = torch.sum(force_z**2)
    magnitude = -1 * b * (mag_x + mag_y + mag_z)


    grad_X = FxsX[0] - FxsX[-1]
    grad_Y = FysY[0] - FysY[-1]
    grad_Z = FzsZ[0] - FzsZ[-1]
    gradient = -1 *c * (grad_X + grad_Y + grad_Z)

    return balance + magnitude + gradient

def levitation_balance_magnitude_grad_fin_diff_greater(force_x, force_y, force_z, weight, torque, **params):
    a,b,c = params["weights"]

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    net_x = torch.sum(force_x)**2
    net_y = torch.sum(force_y)**2
    # counter_weight = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    greater_weight = (weight - torch.sum(force_z) ).unsqueeze_(0)
    balance = a* (net_x + net_y + greater_weight)

    mag_x = torch.sum(force_x**2)
    mag_y = torch.sum(force_y**2)
    mag_z = torch.sum(force_z**2)
    magnitude = -1 * b * (mag_x + mag_y + mag_z)


    grad_X = FxsX[0] - FxsX[-1]
    grad_Y = FysY[0] - FysY[-1]
    grad_Z = FzsZ[0] - FzsZ[-1]
    gradient = -1 *c * (grad_X + grad_Y + grad_Z)

    return balance + magnitude + gradient


def levitation_balance_greater_grad(force_x, force_y, force_z, weight, torque, **params):
    a,b,c = params["weights"]

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

    return balance + greater_weight + gradient


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

def levitation_balance_mag_grad_torque_gerater(force_x, force_y, force_z, weight, torque, **params):
    a,b,c,d,e = params["weights"]
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

    grad_X = FxsX[-1] - FxsX[0]
    grad_Y = 2*(FysY[-1] - FysY[0])
    grad_Z = FzsZ[-1] - FzsZ[0]
    gradient = c * (grad_X + grad_Y + grad_Z)

    min_torque = d*torch.sum(torque**2,dim=[1,2])

    greater_weight = e*(weight - torch.sum(force_z) ).unsqueeze_(0)

    # print(balance, magnitude, gradient, min_torque)
    return balance + magnitude + gradient + min_torque +greater_weight

def levitation_balance_grad_torque_direction(force_x, force_y, force_z, weight, torque, **params):
    a,b,c,d = params["weights"]
    norms = params['norms']

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    net_x = torch.sum(force_x)**2
    net_y = torch.sum(force_y)**2
    net_z = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    balance = a* (net_x + net_y + net_z)

    grad_X = FxsX[-1] - FxsX[0]
    grad_Y = FysY[-1] - FysY[0]
    grad_Z = FzsZ[-1] - FzsZ[0]
    gradient = b * (grad_X + grad_Y + grad_Z)
    
    min_torque = c*torch.sum(torque**2,dim=[1,2])

    force = torch.stack([force_x, force_y, force_z],dim=1)
    alpha = force / norms #should be less than 0
    alpha = alpha.real
    # force_neg = d* torch.sum(torch.maximum(torch.zeros_like(alpha), alpha)**2)
    force_neg=0
    
    # print(balance, gradient, min_torque, force_neg)
    return balance + gradient + min_torque + force_neg


def levitation_balance_grad_torque_direction_greater(force_x, force_y, force_z, weight, torque, **params):
    a,b,c,d,e = params["weights"]
    norms = params['norms']

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    net_x = torch.sum(force_x)**2
    net_y = torch.sum(force_y)**2
    net_z = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    balance = a* (net_x + net_y + net_z)

    grad_X = FxsX[-1] - FxsX[0]
    grad_Y = FysY[-1] - FysY[0]
    grad_Z = FzsZ[-1] - FzsZ[0]
    gradient = b * (grad_X + grad_Y + grad_Z)
    
    min_torque = c*torch.sum(torque**2,dim=[1,2])

    force = torch.stack([force_x, force_y, force_z],dim=1)
    alpha = force / norms #should be less than 0
    alpha = alpha.real
    force_neg = d* torch.sum(torch.maximum(torch.zeros_like(alpha), alpha)**2)

    greater_weight = e*(weight - torch.sum(force_z) ).unsqueeze_(0)
    

    return balance + gradient + min_torque + force_neg + greater_weight


def levitation_balance_grad_torque_direction_greater_sideways(force_x, force_y, force_z, weight, torque, **params):
    a,b,c,d,e = params["weights"]
    norms = params['norms']

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    net_z = torch.sum(force_z)**2
    net_y = torch.sum(force_y)**2
    net_x = ((torch.sum(force_x) + weight)**2).unsqueeze_(0)
    balance = a* (net_x + net_y + net_z)

    grad_X = FxsX[-1] - FxsX[0]
    grad_Y = FysY[-1] - FysY[0]
    grad_Z = FzsZ[-1] - FzsZ[0]
    gradient = b * (grad_X + grad_Y + grad_Z)
    
    min_torque = c*torch.sum(torque**2,dim=[1,2])

    force = torch.stack([force_x, force_y, force_z],dim=1)
    alpha = force / norms #should be less than 0
    alpha = alpha.real
    force_neg = d* torch.sum(torch.maximum(torch.zeros_like(alpha), alpha)**2)

    greater_weight = e*(weight - torch.sum(force_z) ).unsqueeze_(0)
    

    return balance + gradient + min_torque + force_neg + greater_weight

def levitation_balance_grad_torque_sideways(force_x, force_y, force_z, weight, torque, **params):
    a,b,c = params["weights"]

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    net_z = torch.sum(force_z)**2
    net_y = torch.sum(force_y)**2
    net_x = ((torch.sum(force_x) + weight)**2).unsqueeze_(0)
    balance = a* (net_x + net_y + net_z)

    grad_X = FxsX[-1] - FxsX[0]
    grad_Y = FysY[-1] - FysY[0]
    grad_Z = FzsZ[-1] - FzsZ[0]
    gradient = b * (grad_X + grad_Y + grad_Z)
    
    min_torque = c*torch.sum(torque**2,dim=[1,2])
    

    return balance + gradient + min_torque 


def levitation_big_f_grad(force_x, force_y, force_z, weight, torque, **params):
    a,b= params["weights"]
    norms = params['norms']

    FxsX = params["FxsX"]
    FysY = params["FysY"]
    FzsZ = params["FzsZ"]

    net_x = torch.sum(force_x)**2
    net_y = torch.sum(force_y)**2
    # net_z = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
    net_z = (-1*weight) - torch.sum(force_z).unsqueeze_(0)
    balance = a* (net_x + net_y + net_z)

    grad_X = FxsX[-1] - FxsX[0]
    grad_Y = FysY[-1] - FysY[0]
    grad_Z = FzsZ[-1] - FzsZ[0]
    gradient = b * (grad_X + grad_Y + grad_Z)
    
    # print(balance, gradient, min_torque, force_neg)
    return balance + gradient 
