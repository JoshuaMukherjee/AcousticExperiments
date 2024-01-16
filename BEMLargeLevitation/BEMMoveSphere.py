from BEMLevitationObjectives import BEM_levitation_objective_subsample_stability_fin_diff, levitation_balance_greater_grad_torque

from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas,\
      get_centre_of_mass_as_points, get_weight, load_multiple_scatterers, merge_scatterers, get_lines_from_plane,get_plane, translate
from acoustools.Utilities import TRANSDUCERS, write_to_file, get_rows_in, propagate_abs
from acoustools.BEM import grad_H, get_cache_or_compute_H_gradients, get_cache_or_compute_H,propagate_BEM_pressure, get_cache_or_compute_H_2_gradients
from acoustools.Visualiser import Visualise, force_quiver
from acoustools.Solvers import gradient_descent_solver
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Gorkov import force_mesh, get_force_mesh_along_axis

from BEMLevUtils import get_H_for_fin_diffs

import torch, vedo
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":

    START = torch.tensor([[0],[-0.01],[0]])
    END = torch.tensor([[0],[0.01],[0]])
    MOVEMENT_STEPS = 60


    diff = 0.0025
    board = TRANSDUCERS

    wall_paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.085,0.085],rotys=[90,-90]) #Make mesh at 0,0,0

    
    ball_path = "Media/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.02)

    translate(ball, START[0].item(), START[1].item(), START[2].item()) #Move to start
    scatterer = merge_scatterers(ball, walls)

    scatterer_cells = get_centres_as_points(scatterer)
    ball_cells = get_centres_as_points(ball)
    
    mask = get_rows_in(scatterer_cells,ball_cells, expand=False)
    

    print(scatterer)

    centres = get_centres_as_points(scatterer)
    centre_of_mass = get_centre_of_mass_as_points(scatterer)
 
    norms = get_normals_as_points(scatterer)
    
    areas = get_areas(scatterer)
    areas = areas.expand((1,-1,-1))

    Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=True)
    H = get_cache_or_compute_H(scatterer,board,print_lines=True)

    weight = -1*get_weight(ball)

    Hss = []
    Hxss = []
    Hyss = []
    Hzss = []

    SCALE = 100
    startX = torch.tensor([[-1*diff],[0],[0]])/SCALE
    endX = torch.tensor([[diff],[0],[0]])/SCALE
    Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startX, endX, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
    Hss.append(Hs)
    Hxss.append(Hxs)
    Hyss.append(Hys)
    Hzss.append(Hzs)

    startY = torch.tensor([[0],[-1*diff],[0]])/SCALE
    endY = torch.tensor([[0],[diff],[0]])/SCALE
    Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startY, endY, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
    Hss.append(Hs)
    Hxss.append(Hxs)
    Hyss.append(Hys)
    Hzss.append(Hzs)

    startZ = torch.tensor([[0],[0],[-1*diff]])/SCALE
    endZ = torch.tensor([[0],[0],[diff]])/SCALE
    Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startZ, endZ, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
    Hss.append(Hs)
    Hxss.append(Hxs)
    Hyss.append(Hys)
    Hzss.append(Hzs)

    WEIGHTS = [1000,100,10000,0]

    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":weight,
        # "weight":-1*0.00100530964,
        "Hgrad":(Hx, Hy, Hz),
        "H":H,
        "loss":levitation_balance_greater_grad_torque,
        "loss_params":{
            'weights':WEIGHTS
        },
        "indexes":mask.squeeze_(),
        "diff":diff,
        "scatterer_elements":[ball,walls],
        "Hss":Hss,
        "Hxss":Hxss,
        "Hyss":Hyss,
        "Hzss":Hzss
    }


    BASE_LR = 1e-3
    MAX_LR = 1e-2
    EPOCHS = 400

    scheduler = torch.optim.lr_scheduler.CyclicLR
    scheduler_args = {
        "max_lr":MAX_LR,
        "base_lr":BASE_LR,
        "cycle_momentum":False,
        "step_size_up":25
    }

    DX = (START[0].item() -END[0].item())/MOVEMENT_STEPS
    DY = (START[1].item() -END[1].item())/MOVEMENT_STEPS
    DZ = (START[2].item() -END[2].item())/MOVEMENT_STEPS

    forces_x = []
    forces_y = []
    forces_z = []
 
    for i in range(MOVEMENT_STEPS):
        print()
        print(i)

        x = gradient_descent_solver(centres, BEM_levitation_objective_subsample_stability_fin_diff,constrains=constrain_phase_only,objective_params=params,log=True,\
                                    iters=EPOCHS,lr=BASE_LR, optimiser=torch.optim.Adam, board=board,scheduler=scheduler, scheduler_args=scheduler_args)
        

        
        force = force_mesh(x,centres,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
        force_x = force[:,0,:][:,mask]
        force_y = force[:,1,:][:,mask]
        force_z = force[:,2,:][:,mask]

        forces_x.append(torch.sum(force_x))
        forces_y.append(torch.sum(force_y))
        forces_z.append(torch.sum(force_z) + weight)

        #Move to next pos
        translate(ball, DX,DY,DZ) #Move to start
        scatterer = merge_scatterers(ball, walls)

        scatterer_cells = get_centres_as_points(scatterer)
        ball_cells = get_centres_as_points(ball)

        #Recompute Everything
        centres = get_centres_as_points(scatterer)
        centre_of_mass = get_centre_of_mass_as_points(scatterer)
    
        norms = get_normals_as_points(scatterer)
        
        areas = get_areas(scatterer)
        areas = areas.expand((1,-1,-1))

        Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=True)
        H = get_cache_or_compute_H(scatterer,board,print_lines=True)

        Hss = []
        Hxss = []
        Hyss = []
        Hzss = []

        SCALE = 100
        startX = torch.tensor([[-1*diff],[0],[0]])/SCALE
        endX = torch.tensor([[diff],[0],[0]])/SCALE
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startX, endX, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
        Hss.append(Hs)
        Hxss.append(Hxs)
        Hyss.append(Hys)
        Hzss.append(Hzs)

        startY = torch.tensor([[0],[-1*diff],[0]])/SCALE
        endY = torch.tensor([[0],[diff],[0]])/SCALE
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startY, endY, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
        Hss.append(Hs)
        Hxss.append(Hxs)
        Hyss.append(Hys)
        Hzss.append(Hzs)

        startZ = torch.tensor([[0],[0],[-1*diff]])/SCALE
        endZ = torch.tensor([[0],[0],[diff]])/SCALE
        Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startZ, endZ, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False)
        Hss.append(Hs)
        Hxss.append(Hxs)
        Hyss.append(Hys)
        Hzss.append(Hzs)

        params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":weight,
        # "weight":-1*0.00100530964,
        "Hgrad":(Hx, Hy, Hz),
        "H":H,
        "loss":levitation_balance_greater_grad_torque,
        "loss_params":{
            'weights':WEIGHTS
        },
        "indexes":mask.squeeze_(),
        "diff":diff,
        "scatterer_elements":[ball,walls],
        "Hss":Hss,
        "Hxss":Hxss,
        "Hyss":Hyss,
        "Hzss":Hzss
        }

    
        # print(torch.sum((force_z_bottom)).item(), params["weight"], torch.sum(force_z_top).item(), (torch.sum((force_z_bottom)) + params["weight"] + torch.sum(force_z_top)).item() )
        print(torch.sum((force_z)).item(), params["weight"], (torch.sum((force_z)) + params["weight"]).item(), 1000 * (torch.sum((force_z)) + params["weight"]).item()/9.81)
        print(torch.sum(torch.abs(force_x)).item(), torch.sum(force_x).item())
        print(torch.sum(torch.abs(force_y)).item(), torch.sum(force_y).item())
        print(torch.sum(torch.abs(force_z)).item(), torch.sum(force_z).item())
        print()
    
    forces_x = [fx.cpu().detach().numpy() for fx in forces_x]
    forces_y = [fy.cpu().detach().numpy() for fy in forces_y]
    forces_z = [fz.cpu().detach().numpy() for fz in forces_z]

    print(forces_x)

    plt.plot(forces_x, label = '$F_x$')
    plt.plot(forces_y, label = '$F_y$')
    plt.plot(forces_z, label = '$F_z - mg$')
    plt.legend()
    plt.show()