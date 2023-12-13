from BEMLevitationObjectives import BEM_levitation_objective, sum_forces_torque,sum_top_bottom_force_torque, max_magnitude_min_force, BEM_levitation_objective_top_bottom,balance

from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas, get_lines_from_plane, downsample, get_centre_of_mass_as_points
from acoustools.Utilities import TRANSDUCERS, propagate_abs, get_convert_indexes, create_board, device, TOP_BOARD, BOTTOM_BOARD
from acoustools.BEM import compute_H, grad_H, get_cache_or_compute_H_gradients, get_cache_or_compute_H,propagate_BEM_pressure
from acoustools.Visualiser import Visualise
from acoustools.Solvers import gradient_descent_solver, wgs_wrapper, wgs_batch
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Gorkov import force_mesh

import torch, vedo

if __name__ == "__main__":

    path = "Media/Sphere-lam2.stl"
    scatterer = load_scatterer(path,dy=-0.06) #Make mesh at 0,0,0
    
    scale_to_diameter(scatterer,0.04)


    print(scatterer)

    board = TRANSDUCERS

    centres = get_centres_as_points(scatterer)
    centre_of_mass = get_centre_of_mass_as_points(scatterer)
 
    norms = get_normals_as_points(scatterer)
    
    areas = get_areas(scatterer)
    areas = areas.expand((1,-1,-1))

    Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=True)
    H = get_cache_or_compute_H(scatterer,board,print_lines=True)

    # top_board = TOP_BOARD
    # bottom_board = BOTTOM_BOARD

    # Hx_top, Hy_top, Hz_top = get_cache_or_compute_H_gradients(scatterer, board=top_board,print_lines=True )
    # Hx_bottom, Hy_bottom, Hz_bottom = get_cache_or_compute_H_gradients(scatterer, board=bottom_board,print_lines=True)
    
    # H_top = get_cache_or_compute_H(scatterer,top_board,print_lines=True)
    # H_bottom = get_cache_or_compute_H(scatterer,bottom_board,print_lines=True)

    # top_board_idx = (centres[:,2,:] > centre_of_mass[:,2,:]) 

    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":-1*0.0027*9.81,
        "Hgrad":(Hx, Hy, Hz),
        "H":H,
        "loss":balance,
        "loss_params":{
              "weights": [1,1,1,1,1]
        }
    }


    BASE_LR = 1e-2
    # MAX_LR = 1e3
    EPOCHS = 1000

    # scheduler = torch.optim.lr_scheduler.CyclicLR
    # scheduler_args = {
    #     "max_lr":MAX_LR,
    #     "base_lr":BASE_LR,
    #     "cycle_momentum":False,
    #     "step_size_up":100
    # }
    #scheduler=scheduler, scheduler_args=scheduler_args    


    x = gradient_descent_solver(centres, BEM_levitation_objective,constrains=constrain_phase_only,objective_params=params,log=True,\
                                iters=EPOCHS,lr=BASE_LR, optimiser=torch.optim.Adam, board=board)



    # f_top = force_mesh(x[:,256:,:],centres,norms,areas,top_board,grad_H,params,Ax=Hx_top, Ay=Hy_top, Az=Hz_top,F=H_top)
    # f_bottom = force_mesh(x[:,:256,:],centres,norms,areas,bottom_board,grad_H,params,Ax=Hx_bottom, Ay=Hy_bottom, Az=Hz_bottom,F=H_bottom)
    
    # force_x = f_top[:,0,:] + f_bottom[:,0,:]
    # force_y = f_top[:,1,:] + f_bottom[:,1,:]
    # force_z_top =f_top[:,2,:]
    # force_z_bottom = f_bottom[:,2,:]

    force = force_mesh(x,centres,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    force_x = force[:,0,:]
    force_y = force[:,1,:]
    force_z = force[:,2,:]
    
    # print(torch.sum((force_z_bottom)).item(), params["weight"], torch.sum(force_z_top).item(), (torch.sum((force_z_bottom)) + params["weight"] + torch.sum(force_z_top)).item() )
    print(torch.sum((force_z)).item(), params["weight"], (torch.sum((force_z)) + params["weight"]).item() )
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

    # Visualise(A,B,C,x,colour_functions=[propagate_abs], add_lines_functions=[get_lines_from_plane],add_line_args=[line_params],\
    #           colour_function_args=[{}],vmax=9000)
    



    # FLIP_INDEXES = get_convert_indexes()
    # row = torch.angle(x).squeeze_()
    # row_flip = row[FLIP_INDEXES]


    # num_frames = 1
    # num_transducers = 512

    # output_f = open("./BEMLargeLevitation/Paths/spherelev"+".csv","w")
    # output_f.write(str(num_frames)+","+str(num_transducers)+"\n")
    # for i,phase in enumerate(row_flip):
    #             output_f.write(str(phase.item()))
    #             if i < 511:
    #                 output_f.write(",")
    #             else:
    #                 output_f.write("\n")

    # output_f.close()