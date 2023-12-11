from BEMLevitationObjectives import BEM_levitation_objective, sum_forces_torque,sum_top_bottom_force_torque, max_magnitude_min_force
from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas, get_lines_from_plane, downsample, get_centre_of_mass_as_points
from acoustools.Utilities import TRANSDUCERS, propagate_abs, get_convert_indexes
from acoustools.BEM import compute_H, grad_H, get_cache_or_compute_H_gradients, get_cache_or_compute_H,propagate_BEM_pressure
from acoustools.Visualiser import Visualise
from acoustools.Solvers import gradient_descent_solver, wgs_wrapper, wgs_batch
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Gorkov import force_mesh

import torch, vedo

if __name__ == "__main__":

    path = "Media/Sphere-lam2.stl"
    scatterer = load_scatterer(path,dy=-0.06) #Make mesh at 0,0,0
    
    scale_to_diameter(scatterer,0.08)


    print(scatterer)

    board = TRANSDUCERS

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
        "Hgrad":(Hx,Hy,Hz),
        "H":H,
        "loss":max_magnitude_min_force,
        "loss_params":{
              "top_board_idx":top_board_idx,
              "weights": [1,1,1,1,1e-2,1e-1,1e-1]
        }
    }


    BASE_LR = 1e-3
    MAX_LR = 1e-1
    EPOCHS = 4000

    scheduler = torch.optim.lr_scheduler.CyclicLR
    scheduler_args = {
        "max_lr":MAX_LR,
        "base_lr":BASE_LR,
        "cycle_momentum":False,
        "step_size_up":100
    }
    


    x = gradient_descent_solver(centres, BEM_levitation_objective,constrains=constrain_phase_only,objective_params=params,log=True,\
                                iters=EPOCHS,lr=BASE_LR,scheduler=scheduler, scheduler_args=scheduler_args)
    f = force_mesh(x,centres,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    
    force_x = f[:,0,:]
    force_y = f[:,1,:]
    force_z = f[:,2,:]
    force_z_top =force_z[top_board_idx]
    force_z_bottom = force_z[~top_board_idx]
    
    print(torch.sum((force_z_bottom)).item(), params["weight"], torch.sum(force_z_top).item(), (torch.sum((force_z_bottom)) + params["weight"] + torch.sum(force_z_top)).item() )
    print(torch.sum(force_x).item())
    print(torch.sum(force_y).item())


    A = torch.tensor((-0.07,0, 0.07))
    B = torch.tensor((0.07,0, 0.07))
    C = torch.tensor((-0.07,0, -0.07))

    origin = (0,0,-0.07)
    normal = (0,1,0)

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}

   

    Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure], add_lines_functions=[get_lines_from_plane],add_line_args=[line_params],\
              colour_function_args=[{"H":H,"scatterer":scatterer,"board":board}],vmax=9000)
    
    exit()


    FLIP_INDEXES = get_convert_indexes()
    row = torch.angle(x).squeeze_()
    row_flip = row[FLIP_INDEXES]


    num_frames = 1
    num_transducers = 512

    output_f = open("./BEMLargeLevitation/Paths/spherelev"+".csv","w")
    output_f.write(str(num_frames)+","+str(num_transducers)+"\n")
    for i,phase in enumerate(row_flip):
                output_f.write(str(phase.item()))
                if i < 511:
                    output_f.write(",")
                else:
                    output_f.write("\n")

    output_f.close()