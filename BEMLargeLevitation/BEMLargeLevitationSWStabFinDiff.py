from BEMLevitationObjectives import BEM_levitation_objective_subsample_stability_fin_diff, balance_greater_z_stab_fin_diff, levitation_balance_greater_grad_torque

from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas,\
      get_centre_of_mass_as_points, get_weight, load_multiple_scatterers, merge_scatterers, get_lines_from_plane,get_plane
from acoustools.Utilities import TRANSDUCERS, write_to_file, get_rows_in
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

    diff = 0.0025
    board = TRANSDUCERS

    wall_paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.06,0.06],rotys=[90,-90]) #Make mesh at 0,0,0
    
    
    ball_path = "Media/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.02)

    scatterer = merge_scatterers(ball, walls)

    # ball_points = scatterer.vertices[ball_ids]
    # ball_cells = scatterer.map_cells_to_points(ball_points)
    # print(ball_cells)

    

    #Get a mask of the cell faces for the object to be levitated

    scatterer_cells = get_centres_as_points(scatterer)
    ball_cells = get_centres_as_points(ball)
    
    mask = get_rows_in(scatterer_cells,ball_cells, expand=False)
    
    # scale_to_diameter(scatterer,0.04)

    print(scatterer)

    centres = get_centres_as_points(scatterer)
    centre_of_mass = get_centre_of_mass_as_points(scatterer)
 
    norms = get_normals_as_points(scatterer)
    
    areas = get_areas(scatterer)
    areas = areas.expand((1,-1,-1))

    Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=True)
    H = get_cache_or_compute_H(scatterer,board,print_lines=True)
    Haa = get_cache_or_compute_H_2_gradients(scatterer, board,print_lines=True)

    # indexes = get_indexes_subsample(1700, centres)
 
    # weight = -1*0.0027*9.81
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

    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":weight,
        # "weight":-1*0.00100530964,
        "Hgrad":(Hx, Hy, Hz),
        "H":H,
        "Hgrad2":Haa,
        "loss":levitation_balance_greater_grad_torque,
        "loss_params":{
            #   "weights": [1000,1,1,1,1,1,1e-17,1000,10000]
            # "weights": [1000,1,1,1,1,10,10,10,100,100]#ForceXYZFinDiff
            # "weights": [1000,1,1,1,1,20,10,50,10,10]#ForceVisFinDiff
            # "weights": [5000,1,1,1,100,20,20,20,10,10]
            # "weights":[10,1,1] #BMGForceXYZ - levitation_balance_magnitude_grad_fin_diff
            # "weights":[1,1,100] #BMGGreater - levitation_balance_magnitude_grad_fin_diff_greater
            # "weights":[10,10,1] #BGG levitation_balance_greater_grad
            # "weights":[1000,10,10000]#BGG_LargeForce levitation_balance_greater_grad
            'weights':[1000,10,10000,0]
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
    # scheduler=scheduler, scheduler_args=scheduler_args    


    x = gradient_descent_solver(centres, BEM_levitation_objective_subsample_stability_fin_diff,constrains=constrain_phase_only,objective_params=params,log=True,\
                                iters=EPOCHS,lr=BASE_LR, optimiser=torch.optim.Adam, board=board,scheduler=scheduler, scheduler_args=scheduler_args)
    

    
    force = force_mesh(x,centres,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    force_x = force[:,0,:][:,mask]
    force_y = force[:,1,:][:,mask]
    force_z = force[:,2,:][:,mask]


    
    # print(torch.sum((force_z_bottom)).item(), params["weight"], torch.sum(force_z_top).item(), (torch.sum((force_z_bottom)) + params["weight"] + torch.sum(force_z_top)).item() )
    print(torch.sum((force_z)).item(), params["weight"], (torch.sum((force_z)) + params["weight"]).item(), 1000 * (torch.sum((force_z)) + params["weight"]).item()/9.81)
    print(torch.sum(torch.abs(force_x)).item(), torch.sum(force_x).item())
    print(torch.sum(torch.abs(force_y)).item(), torch.sum(force_y).item())
    print(torch.sum(torch.abs(force_z)).item(), torch.sum(force_z).item())
    

    A = torch.tensor((-0.07,0, 0.07))
    B = torch.tensor((0.07,0, 0.07))
    C = torch.tensor((-0.07,0, -0.07))
    normal = (0,1,0)
    origin = (0,0,-0.07)

    # A = torch.tensor((-0.07, 0.07,0))
    # B = torch.tensor((0.07, 0.07,0))
    # C = torch.tensor((-0.07, -0.07,0))
    # normal = (0,0,1)
    # origin = (0,0,0)
    

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}
    line_params_wall = {"scatterer":walls,"origin":origin,"normal":normal}

    # Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure,propagate_BEM_pressure], add_lines_functions=[get_lines_from_plane,get_lines_from_plane],add_line_args=[line_params,line_params],\
            #   colour_function_args=[{"H":H,"scatterer":scatterer,"board":board},{"board":board,"scatterer":walls}],vmax=9000, show=True)
   
   

    # write_to_file(x,"./BEMLargeLevitation/Paths/spherelev.csv",1)


    pad = 0.005
    planar = get_plane(scatterer,origin,normal)
    bounds = ball.bounds()
    xlim=[bounds[0]-pad,bounds[1]+pad]
    ylim=[bounds[2]-pad,bounds[3]+pad]

    norms = get_normals_as_points(ball)
    # force_quiver(centres[:,:,mask],norms[:,0,:],norms[:,2,:], normal,xlim,ylim,show=False,log=False)
    force_quiver(centres[:,:,mask],force_x,force_z, normal,xlim,ylim,show=False,log=False)
    plt.show()
    # exit()
    
    startX = torch.tensor([[-1*diff],[0],[0]])
    endX = torch.tensor([[diff],[0],[0]])

    startY = torch.tensor([[0],[-1*diff],[0]])
    endY = torch.tensor([[0],[diff],[0]])

    startZ = torch.tensor([[0],[0],[-1*diff]])
    endZ = torch.tensor([[0],[0],[diff]])
    
    
    
    steps = 60
    path = "Media"
    FxsX, FysX, FzsX = get_force_mesh_along_axis(startX, endX, x, [ball.clone(),walls], board,mask,steps=steps, use_cache=True, print_lines=False,path=path)
    FxsY, FysY, FzsY = get_force_mesh_along_axis(startY, endY, x, [ball.clone(),walls], board,mask,steps=steps, use_cache=True, print_lines=False,path=path)
    FxsZ, FysZ, FzsZ = get_force_mesh_along_axis(startZ, endZ, x, [ball.clone(),walls], board,mask,steps=steps, use_cache=True, print_lines=False,path=path)

    for i,(Fxs, Fys, Fzs) in enumerate([[FxsX, FysX, FzsX], [FxsY, FysY, FzsY], [FxsZ, FysZ, FzsZ] ]):
        Fxs = [f.cpu().detach().numpy() for f in Fxs]
        Fys = [f.cpu().detach().numpy() for f in Fys]
        Fzs = [f.cpu().detach().numpy() + weight for f in Fzs]
        

        xticklabs = [-1* diff, 0 , diff]
        xticks = [0, steps/2 , steps]
        
        plt.subplot(3,1,i+1)
        plt.plot(Fxs, label="$F_x$")
        plt.plot(Fys, label="$F_y$")
        plt.plot(Fzs, label="$F_z-mg$")
        plt.xlabel("Distance (mm)")
        plt.xticks(xticks, xticklabs)
        plt.ylabel("Restoring Force")
    plt.legend()
    plt.show()



    