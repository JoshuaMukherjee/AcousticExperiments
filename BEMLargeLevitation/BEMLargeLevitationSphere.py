from BEMLevitationObjectives import BEM_levitation_objective, sum_forces_torque,sum_top_bottom_force_torque, max_magnitude_min_force,\
     BEM_levitation_objective_top_bottom,balance, BEM_levitation_objective_subsample, balance_max_z, weight_force, balance_greater_z_stability_equal, BEM_levitation_objective_subsample_stability
from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas, get_lines_from_plane, downsample,\
      get_centre_of_mass_as_points, get_weight
from acoustools.Utilities import TRANSDUCERS, propagate_abs, get_convert_indexes, create_board, device, TOP_BOARD, BOTTOM_BOARD, write_to_file, get_rows_in
from acoustools.BEM import compute_H, grad_H, get_cache_or_compute_H_gradients, get_cache_or_compute_H,propagate_BEM_pressure, get_cache_or_compute_H_2_gradients
from acoustools.Visualiser import Visualise
from acoustools.Solvers import gradient_descent_solver, wgs_wrapper, wgs_batch
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Gorkov import force_mesh, get_force_mesh_along_axis

from BEMLevUtils import get_indexes_subsample

import torch, vedo
import matplotlib.pyplot as plt

if __name__ == "__main__":

    path = "Media/Sphere-lam2.stl"
    scatterer = load_scatterer(path,dy=-0.06) #Make mesh at 0,0,0
    
    scale_to_diameter(scatterer,0.02)


    print(scatterer)

    board = TRANSDUCERS

    centres = get_centres_as_points(scatterer)
    centre_of_mass = get_centre_of_mass_as_points(scatterer)
 
    norms = get_normals_as_points(scatterer)
    
    areas = get_areas(scatterer)
    areas = areas.expand((1,-1,-1))

    Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=True)
    H = get_cache_or_compute_H(scatterer,board,print_lines=True)
    Haa = get_cache_or_compute_H_2_gradients(scatterer, board,print_lines=True)

    # indexes = get_indexes_subsample(1700, centres)
    mask = get_rows_in(centres, centres, expand=False)

    weight = -1*0.0027*9.81
    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":weight,
        # "weight":-1*0.00100530964,
        "Hgrad":(Hx, Hy, Hz),
        "H":H,
        "Hgrad2":Haa,
        "loss":balance_greater_z_stability_equal,
        "loss_params":{
            #   "weights": [1000,1,1,1,1,1,1e-17,1000,10000]
            "weights": [1000,1,1,1,1,1,1e-20,100,50]
        },
        "indexes":mask.squeeze()
    }


    BASE_LR = 1e-2
    MAX_LR = 1e-1
    EPOCHS = 200

    scheduler = torch.optim.lr_scheduler.CyclicLR
    scheduler_args = {
        "max_lr":MAX_LR,
        "base_lr":BASE_LR,
        "cycle_momentum":False,
        "step_size_up":100
    }
    # scheduler=scheduler, scheduler_args=scheduler_args    


    x = gradient_descent_solver(centres, BEM_levitation_objective_subsample_stability,constrains=constrain_phase_only,objective_params=params,log=True,\
                                iters=EPOCHS,lr=BASE_LR, optimiser=torch.optim.Adam, board=board,scheduler=scheduler, scheduler_args=scheduler_args)


    force = force_mesh(x,centres,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    force_x = force[:,0,:]
    force_y = force[:,1,:]
    force_z = force[:,2,:]
    
    # print(torch.sum((force_z_bottom)).item(), params["weight"], torch.sum(force_z_top).item(), (torch.sum((force_z_bottom)) + params["weight"] + torch.sum(force_z_top)).item() )
    print(torch.sum((force_z)).item(), params["weight"], (torch.sum((force_z)) + params["weight"]).item(), 1000 * (torch.sum((force_z)) + params["weight"]).item()/9.81)
    print(torch.sum(torch.abs(force_z)).item())
    print(torch.sum(force_x).item())
    print(torch.sum(force_y).item())

    A = torch.tensor((-0.07,0, 0.07))
    B = torch.tensor((0.07,0, 0.07))
    C = torch.tensor((-0.07,0, -0.07))

    origin = (0,0,-0.07)
    normal = (0,1,0)

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}

    Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure,propagate_abs], add_lines_functions=[get_lines_from_plane,None],add_line_args=[line_params,{}],\
              colour_function_args=[{"H":H,"scatterer":scatterer,"board":board},{"board":board}],vmax=9000)
    

    # write_to_file(x,"./BEMLargeLevitation/Paths/spherelev.csv",1)


    