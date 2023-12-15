from BEMLevitationObjectives import BEM_levitation_objective, sum_forces_torque,sum_top_bottom_force_torque, max_magnitude_min_force, balance, balance_max_z,balance_greater_z
from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas,\
      get_lines_from_plane, downsample, get_centre_of_mass_as_points, get_weight, get_plane, load_multiple_scatterers, merge_scatterers, get_mesh_subset_mask
from acoustools.Utilities import TRANSDUCERS, propagate_abs, get_convert_indexes
from acoustools.BEM import compute_H, grad_H, get_cache_or_compute_H_gradients, get_cache_or_compute_H,propagate_BEM_pressure
from acoustools.Visualiser import Visualise, force_quiver
from acoustools.Solvers import gradient_descent_solver, wgs_wrapper, wgs_batch
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Gorkov import force_mesh


from acoustools.Solvers import wgs_wrapper, wgs_batch

import torch, vedo

import matplotlib.pyplot as plt

if __name__ == "__main__":
    board = TRANSDUCERS


    wall_paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.06,0.06],rotys=[90,-90], board=board) #Make mesh at 0,0,0
    # walls.reverse(cells=False,normals=True)

    
    ball_path = "Media/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.04)

    scatterer = merge_scatterers(ball, walls)

    scatterer_cells = get_centres_as_points(scatterer)
    ball_cells = get_centres_as_points(ball)

    mask = get_mesh_subset_mask(ball_cells, scatterer_cells)

    print(scatterer)


    centres = get_centres_as_points(scatterer)
    centre_of_mass = get_centre_of_mass_as_points(scatterer)
 
    norms = get_normals_as_points(scatterer)
    
    areas = get_areas(scatterer)
    areas = areas.expand((1,-1,-1))

    Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=True)
    H = get_cache_or_compute_H(scatterer,board,print_lines=True)

    top_board_idx = (centres[:,2,:] > centre_of_mass[:,2,:]) 

    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":-1*0.0027*9.81,
        # "weight":-1*0.00100530964,
        "Hgrad":(Hx, Hy, Hz),
        "H":H,
        "loss":balance_greater_z,
        "loss_params":{
              "weights": [1000,1,1,1,10,10]
        },
        "indexes":mask.squeeze_()
    }


    BASE_LR = 1e-2
    MAX_LR = 1e-1
    EPOCHS = 10

    scheduler = torch.optim.lr_scheduler.CyclicLR
    scheduler_args = {
        "max_lr":MAX_LR,
        "base_lr":BASE_LR,
        "cycle_momentum":False,
        "step_size_up":100
    }
    


    x = gradient_descent_solver(centres, BEM_levitation_objective,constrains=constrain_phase_only,objective_params=params,log=True,\
                                iters=EPOCHS,lr=BASE_LR,scheduler=scheduler, scheduler_args=scheduler_args)
    
    force = force_mesh(x,centres,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    force_x = force[:,0,:]
    force_y = force[:,1,:]
    force_z = force[:,2,:]


    
    # print(torch.sum((force_z_bottom)).item(), params["weight"], torch.sum(force_z_top).item(), (torch.sum((force_z_bottom)) + params["weight"] + torch.sum(force_z_top)).item() )
    print(torch.sum((force_z)).item(), params["weight"], (torch.sum((force_z)) + params["weight"]).item(), 1000 * (torch.sum((force_z)) + params["weight"]).item()/9.81)
    print(torch.sum(torch.abs(force_z)).item())
    print(torch.sum(force_x).item())
    print(torch.sum(force_y).item())
    
    origin = (0,0,0)
    normal = (0,1,0) 

    planar = get_plane(scatterer,origin,normal)
    bounds = planar.bounds()
    

    centres = get_centres_as_points(planar)
    centre_of_mass = get_centre_of_mass_as_points(planar)

 
    norms = get_normals_as_points(planar)
    
    areas = get_areas(planar)
    areas = areas.expand((1,-1,-1))

    Hx, Hy, Hz = get_cache_or_compute_H_gradients(planar, board,print_lines=True)
    H = get_cache_or_compute_H(planar,board,print_lines=True)


    # f = force_mesh(x,centres,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)

    # force_x = f[:,0,:]
    # force_y = f[:,1,:]
    # force_z = f[:,2,:]

    mask = get_mesh_subset_mask(centres, scatterer_cells)


    A = torch.tensor((-0.07,0, 0.07))
    B = torch.tensor((0.07,0, 0.07))
    C = torch.tensor((-0.07,0, -0.07))


    pad = 0.005
    xlim=[bounds[0]-pad,bounds[1]+pad]
    ylim=[bounds[2]-pad,bounds[3]+pad]

    print(centres.shape, force_x[mask].shape)

    force_quiver(centres,force_x[mask],force_z[mask], normal,xlim,ylim,show=False,log=True)
    force_quiver(centres,norms[:,0,:],norms[:,2,:], normal,xlim,ylim,colour="orange",show=False)
    plt.show()
    
    
