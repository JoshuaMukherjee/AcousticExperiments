from BEMLevitationObjectives import BEM_levitation_objective_subsample_stability_fin_diff, levitation_balance_grad_torque_direction,\
      levitation_balance_grad_torque_direction_greater, levitation_balance_mag_grad_torque,\
          levitation_balance_mag_grad_torque_gerater, levitation_big_f_grad

from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas,\
      get_centre_of_mass_as_points, get_weight, load_multiple_scatterers, merge_scatterers, get_lines_from_plane,get_plane, scatterer_file_name, get_edge_data
from acoustools.Utilities import TRANSDUCERS, write_to_file, get_rows_in, propagate_abs, device, DTYPE
from acoustools.BEM import grad_H, get_cache_or_compute_H_gradients, get_cache_or_compute_H,propagate_BEM_pressure, get_cache_or_compute_H_2_gradients, compute_G
from acoustools.Visualiser import Visualise, force_quiver, force_quiver_3d, Visualise_mesh, ABC
from acoustools.Solvers import gradient_descent_solver, wgs
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Force import force_mesh, get_force_mesh_along_axis,torque_mesh
import acoustools.Constants as Constants

from BEMLevUtils import get_H_for_fin_diffs

import torch, vedo
import numpy as np
import pickle

import matplotlib.pyplot as plt

if __name__ == "__main__":

    diff = 0.0025
    board = TRANSDUCERS

    '''
    Load meshes and compute values 
    '''
 
    wall_paths = ["Media/flat-lam2.stl","Media/flat-lam2.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.198/2,0.198/2],rotys=[90,-90]) #Make mesh at 0,0,0
    walls.scale((1,19.3/12,22.5/12),reset=True,origin =False)
    # print(walls)
    walls.filename = scatterer_file_name(walls)
    # print(walls)
    get_edge_data(walls)
    


    ball_path = "Media/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.02)
    
    get_edge_data(ball)
    # scale_to_diameter(ball, Constants.R*2)

    scatterer = merge_scatterers(ball, walls)

    # vedo.show(scatterer,axes=1)
    # exit()

    # ball_points = scatterer.vertices[ball_ids]
    # ball_cells = scatterer.map_cells_to_points(ball_points)
    # print(ball_cells)

    

    #Get a mask of the cell faces for the object to be levitated

    scatterer_cells = get_centres_as_points(scatterer)
    ball_cells = get_centres_as_points(ball)
    
    mask = get_rows_in(scatterer_cells,ball_cells, expand=False)
    
    # scale_to_diameter(scatterer,0.04)

    # print(scatterer)

    centres = get_centres_as_points(scatterer)
    centre_of_mass = get_centre_of_mass_as_points(scatterer)
 
    norms = get_normals_as_points(scatterer)
    
    areas = get_areas(scatterer)
    areas = areas.expand((1,-1,-1))

    Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=True)
    H = get_cache_or_compute_H(scatterer,board,print_lines=True)
    H_Walls = get_cache_or_compute_H(walls,board,print_lines=True)


    # indexes = get_indexes_subsample(1700, centres)
 
    # weight = -1*0.0027*9.81
    # weight = -1*get_weight(ball)
    weight = -1*(0.1/1000)*9.81 #Measured value
    # weight = -1*0.002*9.81

    Hss = []
    Hxss = []
    Hyss = []
    Hzss = []

    SCALE = 10
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
    
    '''
    Set parameters for optimisation
    '''

    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":weight,
        # "weight":-1*0.00100530964,
        "Hgrad":(Hx, Hy, Hz),
        "H":H,
        # "loss":levitation_balance_grad_torque_direction,
        "loss":levitation_big_f_grad,
        "loss_params":{
            "norms":norms[:,:,mask.squeeze()],
            # 'weights':[40,5,1,50000] 
            'weights':[1e4,1] 
        },
        "indexes":mask.squeeze_(),
        "diff":diff,
        "scatterer_elements":[ball,walls],
        "Hss":Hss,
        "Hxss":Hxss,
        "Hyss":Hyss,
        "Hzss":Hzss
    }

    params_stage_2 = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":weight,
        # "weight":-1*0.00100530964,
        "Hgrad":(Hx, Hy, Hz),
        "H":H,
        # "loss":levitation_balance_grad_torque_direction,
        "loss":levitation_balance_grad_torque_direction,
        "loss_params":{
            "norms":norms[:,:,mask.squeeze()],
            # 'weights':[40,5,1,50000] 
            'weights':[5e2,3,0,0] 
        },
        "indexes":mask.squeeze_(),
        "diff":diff,
        "scatterer_elements":[ball,walls],
        "Hss":Hss,
        "Hxss":Hxss,
        "Hyss":Hyss,
        "Hzss":Hzss
    }


    BASE_LR = 1e2
    MAX_LR = 1e3
    EPOCHS = 50

    scheduler = torch.optim.lr_scheduler.CyclicLR
    scheduler_args = {
        "max_lr":MAX_LR,
        "base_lr":BASE_LR,
        "cycle_momentum":False,
        "step_size_up":25
    }


    BASE_LR2 = 1e-1
    MAX_LR2 = 1
    EPOCHS2 = 100

    scheduler_args2 = {
        "max_lr":MAX_LR2,
        "base_lr":BASE_LR2,
        "cycle_momentum":False,
        "step_size_up":25
    }
    # scheduler=scheduler, scheduler_args=scheduler_args    

    '''
    Initialise with WGS below COM - ignore all scattering etc as just a starting point
    '''
    
    p0 = (centre_of_mass[:,0,:].cpu().detach().item(),centre_of_mass[:,1,:].cpu().detach().item(),centre_of_mass[:,2,:].cpu().detach().item())
    p1 = (centre_of_mass[:,0,:].cpu().detach().item(),centre_of_mass[:,1,:].cpu().detach().item(),-100)
    cell = scatterer.find_cells_along_line(p0,p1,tol=1e-7)[0]
    below_COM = torch.tensor(scatterer.coordinates[scatterer.cells[cell][0]]).unsqueeze_(0).unsqueeze_(2).to(device).to(DTYPE)
    
    below_COM_1 = below_COM.clone()
    below_COM_1[:,0,:] -= 0.005

    below_COM_2 = below_COM.clone()
    below_COM_2[:,0,:] += 0.005

    below_COM = torch.cat([below_COM_1, below_COM_2], dim=2)

    x_start = wgs(below_COM,iter=5)

    '''
    Phase retrieval
    '''

    save_set_n = [n-1 for n in [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,100,150,200]]
    # save_set_n = [n-1 for n in [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,100]]
    compute= False
    if compute:
        x1, loss, result = gradient_descent_solver(centres, BEM_levitation_objective_subsample_stability_fin_diff,constrains=constrain_phase_only,objective_params=params,log=True,\
                                    iters=EPOCHS,lr=BASE_LR, board=board,scheduler=scheduler, scheduler_args=scheduler_args, start=x_start, return_loss=True, save_set_n=save_set_n )
        
        x2, loss, result = gradient_descent_solver(centres, BEM_levitation_objective_subsample_stability_fin_diff,constrains=constrain_phase_only,objective_params=params_stage_2,log=True,\
                                    iters=EPOCHS,lr=BASE_LR2, board=board,scheduler=scheduler, scheduler_args=scheduler_args2, start=x1, return_loss=True, save_set_n=save_set_n )
        
        x, loss, result = gradient_descent_solver(centres, BEM_levitation_objective_subsample_stability_fin_diff,constrains=constrain_phase_only,objective_params=params,log=True,\
                                    iters=5,lr=1e-2, board=board, start=x2, return_loss=True, save_set_n=save_set_n )
        
        print('Logging Reuslts...')
        pickle.dump((loss,result),open('Media/SavedResults/SphereLev.pth','wb'))

        print("Writing Phases...")
        # write_to_file(x,"./BEMLargeLevitation/Paths/spherelev.csv",1)
        pickle.dump(x, open('./BEMLargeLevitation/Paths/holo.pth','wb'))
        print("File Written")
    else:
        x = pickle.load(open('./BEMLargeLevitation/Paths/holo.pth','rb'))
    
    force = force_mesh(x,centres,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    torque = torque_mesh(x,centres,norms,areas,centre_of_mass,board,grad_function=grad_H,grad_function_args=params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    
    force_x = force[:,0,:][:,mask]
    force_y = force[:,1,:][:,mask]
    force_z = force[:,2,:][:,mask]

    torque_x = torque[:,0,:][:,mask]
    torque_y = torque[:,1,:][:,mask]
    torque_z = torque[:,2,:][:,mask]

    '''
    Evaluate solution
    '''

    
    # print(torch.sum((force_z_bottom)).item(), params["weight"], torch.sum(force_z_top).item(), (torch.sum((force_z_bottom)) + params["weight"] + torch.sum(force_z_top)).item() )
    print(torch.sum((force_z)).item(), params["weight"], (torch.sum((force_z)) + params["weight"]).item(), 1000 * (torch.sum((force_z)) + params["weight"]).item()/9.81)
    print(torch.sum(torch.abs(force_x)).item(), torch.sum(force_x).item(), torch.sum(torque_x).item())
    print(torch.sum(torch.abs(force_y)).item(), torch.sum(force_y).item(), torch.sum(torque_y).item())
    print(torch.sum(torch.abs(force_z)).item(), torch.sum(force_z).item() + params["weight"], torch.sum(torque_z).item())
    
   
    # exit()

    # A = torch.tensor((0,-0.09, 0.09)).to(device)
    # B = torch.tensor((0,0.09, 0.09)).to(device)
    # C = torch.tensor((0,-0.09, -0.09)).to(device)
    # normal = (1,0,0)
    # origin = (0,0,0)

    A,B,C = ABC(0.02, plane='xy')
    normal = (0,1,0)
    origin = (0,0,0)

    # A = torch.tensor((-0.07, 0.07,0))
    # B = torch.tensor((0.07, 0.07,0))
    # C = torch.tensor((-0.07, -0.07,0))
    # normal = (0,0,1)
    # origin = (0,0,0)
    

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}
    line_params_wall = {"scatterer":walls,"origin":origin,"normal":normal}

    # Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure,propagate_BEM_pressure], add_lines_functions=[get_lines_from_plane,get_lines_from_plane],add_line_args=[line_params,line_params_wall],\
            #   colour_function_args=[{"H":H,"scatterer":scatterer,"board":board},{"board":board,"scatterer":walls}],vmax=6000, show=True)
# 
    # Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure],
            #   colour_function_args=[{"H":H,"scatterer":scatterer,"board":board},{"board":board,"scatterer":walls}], show=True, res=(600,600))
    
    # Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure],
            #   colour_function_args=[{"H":H,"scatterer":scatterer,"board":board}], show=True, res=(200,200))
    # exit()

    # def GH(activations, **params):
    #     G = compute_G(params['points'], walls).to(torch.complex64)
    #     return torch.abs((G@H_Walls)@activations)

    # Visualise(A,B,C,x,colour_functions=[ GH, propagate_abs ] ,depth=2, res=(600,600), titles=['GH Contribution', 'F Contribution'])

    # exit()
    H_ball = get_cache_or_compute_H(ball, board)
    Visualise_mesh(ball,torch.abs(H_ball@x), clamp=True, vmax=4000, vmin=0, show=True)
    exit()
    
    

    pad = 0.005
    planar = get_plane(scatterer,origin,normal)
    bounds = ball.bounds()
    xlim=[bounds[0]-pad,bounds[1]+pad]
    ylim=[bounds[2]-pad,bounds[3]+pad]

    norms = get_normals_as_points(ball)
    # force_quiver(centres[:,:,mask],norms[:,0,:],norms[:,2,:], normal,xlim,ylim,show=False,log=False)
    # force_quiver(centres[:,:,mask],force_x,force_z, normal,xlim,ylim,show=False,log=False)
    ax = force_quiver_3d(centres[:,:,mask]*1.1, force_x, force_y, force_z, scale=100)


    plt.show()

    exit()
    
    startX = torch.tensor([[-1*diff],[0],[0]])
    endX = torch.tensor([[diff],[0],[0]])

    startY = torch.tensor([[0],[-1*diff],[0]])
    endY = torch.tensor([[0],[diff],[0]])

    startZ = torch.tensor([[0],[0],[-1*diff]])
    endZ = torch.tensor([[0],[0],[diff]])
    
    # exit()
    
    steps = 10
    path = "Media"
    print_lines = False
    FxsX, FysX, FzsX = get_force_mesh_along_axis(startX, endX, x, [ball.clone(),walls], board,mask,steps=steps, use_cache=True, print_lines=print_lines,path=path)
    FxsY, FysY, FzsY = get_force_mesh_along_axis(startY, endY, x, [ball.clone(),walls], board,mask,steps=steps, use_cache=True, print_lines=print_lines,path=path)
    FxsZ, FysZ, FzsZ = get_force_mesh_along_axis(startZ, endZ, x, [ball.clone(),walls], board,mask,steps=steps, use_cache=True, print_lines=print_lines,path=path)


    BUFFER_SCALE = 1.2 

    xmax = torch.max(torch.tensor([[FxsX, FxsY, FxsZ]]) )
    ymax = torch.max(torch.tensor([[FysX, FysY, FysZ]]) )
    zmax = torch.max(torch.tensor([[FzsX, FzsY, FzsZ]]) )
    ytickmax = max([xmax, ymax,zmax]) * BUFFER_SCALE

    xmin = torch.min(torch.tensor([[FxsX, FxsY, FxsZ]]) )
    ymin = torch.min(torch.tensor([[FysX, FysY, FysZ]]) )
    zmin = torch.min(torch.tensor([[FzsX, FzsY, FzsZ]]) )
    ytickmin = min([xmin, ymin,zmin]) * BUFFER_SCALE

    labs = ["X", "Y", "Z"]
    for i,(Fxs, Fys, Fzs) in enumerate([[FxsX, FysX, FzsX], [FxsY, FysY, FzsY], [FxsZ, FysZ, FzsZ] ]):
        Fxs = [f.cpu().detach().numpy() for f in Fxs]
        Fys = [f.cpu().detach().numpy() for f in Fys]
        Fzs = [f.cpu().detach().numpy() + weight for f in Fzs]
        

        # xticklabs = [-1* diff, 0 , diff]
        # xticks = [0, steps/2 , steps]

        xticks = torch.linspace(-1*diff, diff, steps+1)
        
        plt.subplot(3,1,i+1)
        plt.plot(xticks,Fxs, label="$F_x$")
        plt.plot(xticks,Fys, label="$F_y$")
        plt.plot(xticks,Fzs, label="$F_z-mg$")
        plt.xlabel("$\Delta$" + labs[i] + " (m)")
        # plt.xticks(xticks, xticklabs)
        plt.ylim(ytickmin,ytickmax)
        plt.ylabel("Force (N)")
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.show()




    