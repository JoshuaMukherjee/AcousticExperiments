from acoustools.Utilities import TRANSDUCERS, get_rows_in, add_lev_sig, write_to_file
from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, load_scatterer, scale_to_diameter, merge_scatterers, get_centres_as_points, get_centre_of_mass_as_points, get_normals_as_points, get_areas, get_lines_from_plane, get_plane
from acoustools.BEM import compute_E, BEM_forward_model_grad, propagate_BEM_pressure
from acoustools.Solvers import gradient_descent_solver
from acoustools.Visualiser import Visualise, force_quiver, force_quiver_3d
from acoustools.Force import force_mesh, torque_mesh, get_force_mesh_along_axis

from BEMLevUtils import get_E_for_fin_diffs
from BEMLevitationObjectives_E import levitation_balance_greater_grad_torque, BEM_levitation_objective_subsample_stability_fin_diff_E, balance_sign_mag, BEM_E_pressure_objective, pressure_direction_loss, levitation_balance_grad_torque_direction_greater
from acoustools.Optimise.Constraints import constrain_phase_only


import torch, vedo
import matplotlib.pyplot as plt


if __name__ == "__main__":

    diff = 0.0025
    board = TRANSDUCERS
 
    wall_paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.175/2,0.175/2],rotys=[90,-90]) #Make mesh at 0,0,0
    walls.scale((1,19/12,19/12),reset=True,origin =False)
    walls.filename = scatterer_file_name(walls)
    # print(walls)


    ball_path = "Media/sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    print(ball)
    scale_to_diameter(ball,0.02)
    # scale_to_diameter(ball, Constants.R*2)

    scatterer = merge_scatterers(ball, walls)

    scatterer_cells = get_centres_as_points(scatterer)
    ball_cells = get_centres_as_points(ball)
    
    mask = get_rows_in(scatterer_cells,ball_cells, expand=False)


    NORMAL_SCALE = 0.001
    centres = get_centres_as_points(scatterer, add_normals=True, normal_scale=NORMAL_SCALE)
    centre_of_mass = get_centre_of_mass_as_points(scatterer)

    norms = get_normals_as_points(scatterer)
    
    areas = get_areas(scatterer)
    areas = areas.expand((1,-1,-1))
    
    E,F,G,H = compute_E(scatterer, centres, TRANSDUCERS, return_components=True)
    Ex, Ey, Ez = BEM_forward_model_grad(centres, scatterer, TRANSDUCERS)
    
    Ess = []
    Exss = []
    Eyss = []
    Ezss = []
    
    SCALE = 10
    
    startX = torch.tensor([[-1*diff],[0],[0]])/SCALE
    endX = torch.tensor([[diff],[0],[0]])/SCALE
    Es, Exs, Eys, Ezs = get_E_for_fin_diffs(startX, endX, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False,normal_scale=NORMAL_SCALE)
    Ess.append(Es)
    Exss.append(Exs)
    Eyss.append(Eys)
    Ezss.append(Ezs)

    startY = torch.tensor([[0],[-1*diff],[0]])/SCALE
    endY = torch.tensor([[0],[diff],[0]])/SCALE
    Es, Exs, Eys, Ezs = get_E_for_fin_diffs(startY, endY, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False,normal_scale=NORMAL_SCALE)
    Ess.append(Es)
    Exss.append(Exs)
    Eyss.append(Eys)
    Ezss.append(Ezs)

    startZ = torch.tensor([[0],[0],[-1*diff]])/SCALE
    endZ = torch.tensor([[0],[0],[diff]])/SCALE
    Hs, Hxs, Hys, Hzs = get_E_for_fin_diffs(startZ, endZ, [ball.clone(),walls], board, steps=1, use_cache=True, print_lines=False,normal_scale=NORMAL_SCALE)
    Ess.append(Es)
    Exss.append(Exs)
    Eyss.append(Eys)
    Ezss.append(Ezs)


    weight = -1*(0.1/1000)*9.81 #Measured value


    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":weight,
        # "weight":-1*0.00100530964,
        "EGrad":(Ex, Ey, Ez),
        "E":E,
        "loss":levitation_balance_grad_torque_direction_greater,
        "loss_params":{
            # 'weights':[1e5,1,1e-5,1],
            "norms":norms[:,:,mask.squeeze()],
            'weights':[100,10,1,0,1e-1] 
        },
        "indexes":mask.squeeze_(),
        "diff":diff,
        "scatterer_elements":[ball,walls],
        "Ess":Ess,
        "Exss":Exss,
        "Eyss":Eyss,
        "Ezss":Ezss
    }

    BASE_LR = 1e-4
    MAX_LR = 1e-2
    EPOCHS = 500

    scheduler = torch.optim.lr_scheduler.CyclicLR
    scheduler_args = {
        "max_lr":MAX_LR,
        "base_lr":BASE_LR,
        "cycle_momentum":False,
        "step_size_up":25
    }
    # scheduler=scheduler, scheduler_args=scheduler_args    


    x = gradient_descent_solver(centres, BEM_levitation_objective_subsample_stability_fin_diff_E,constrains=constrain_phase_only,objective_params=params,log=True,\
                                iters=EPOCHS,lr=BASE_LR, optimiser=torch.optim.Adam, board=board,scheduler=scheduler, scheduler_args=scheduler_args)

    
    # # ME,i = torch.max(torch.abs(E@x),dim=1)
    # # MGH, j = torch.max(torch.abs(G@H@x),dim=1)
    # # Pf = torch.abs(F@x)[:,i]
    # # print(i==j, ME + Pf, MGH)
    # # x = x.to(torch.complex128)
    # print(torch.sum((F+G@H)@x == E@x))
    # print(torch.sum(F@x+(G@H)@x == E@x))
    # print(torch.sum(torch.isclose((F@x+(G@H)@x), E@x)))

    # exit()

    # x = add_lev_sig(x)

    force = force_mesh(x,centres,norms,areas,board,params,Ax=Ex, Ay=Ey, Az=Ez,F=E)
    torque = torque_mesh(x,centres,norms,areas,centre_of_mass,board,force=force)
    
    force_x = force[:,0,:][:,mask] 
    force_y = force[:,1,:][:,mask]
    force_z = force[:,2,:][:,mask]


    torque_x = torque[:,0,:][:,mask]
    torque_y = torque[:,1,:][:,mask]
    torque_z = torque[:,2,:][:,mask]

    # print(torch.sum((force_z_bottom)).item(), params["weight"], torch.sum(force_z_top).item(), (torch.sum((force_z_bottom)) + params["weight"] + torch.sum(force_z_top)).item() )
    print(torch.sum((force_z)).item(), params["weight"], (torch.sum((force_z)) + params["weight"]).item(), 1000 * (torch.sum((force_z)) + params["weight"]).item()/9.81)
    print(torch.sum(torch.abs(force_x)).item(), torch.sum(force_x).item(), torch.sum(torque_x).item())
    print(torch.sum(torch.abs(force_y)).item(), torch.sum(force_y).item(), torch.sum(torque_y).item())
    print(torch.sum(torch.abs(force_z)).item(), torch.sum(force_z).item() + params["weight"], torch.sum(torque_z).item())

    # exit()

    print("Writing...")
    write_to_file(x,"./BEMLargeLevitation/Paths/spherelev.csv",1)
    print("File Written")

    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    # A = torch.tensor((-0.01,0, 0.01))
    # B = torch.tensor((0.01,0, 0.01))
    # C = torch.tensor((-0.01,0, -0.01))

    # A = torch.tensor((-0.07, 0.07,0))
    # B = torch.tensor((0.07, 0.07,0))
    # C = torch.tensor((-0.07, -0.07,0))
    # normal = (0,0,1)
    # origin = (0,0,0)
    

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}
    line_params_wall = {"scatterer":walls,"origin":origin,"normal":normal}
    Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure,propagate_BEM_pressure], add_lines_functions=[get_lines_from_plane,get_lines_from_plane],add_line_args=[line_params,line_params_wall],\
              colour_function_args=[{"H":H,"scatterer":scatterer,"board":board},{"board":board,"scatterer":walls}],vmax=9000, show=True)
    

    pad = 0.005
    planar = get_plane(scatterer,origin,normal)
    bounds = ball.bounds()
    xlim=[bounds[0]-pad,bounds[1]+pad]
    ylim=[bounds[2]-pad,bounds[3]+pad]

    norms = get_normals_as_points(ball)
    # force_quiver(centres[:,:,mask],norms.real[:,0,:],norms.real[:,2,:], normal,xlim,ylim,show=False,log=False,colour='red')
    # force_quiver(centres[:,:,mask].real,force_x,force_z, normal,xlim,ylim,show=False,log=False)
    # plt.show()

    # ax = plt.figure().add_subplot(projection='3d')
    # scale = 1
    # ax.quiver(centres[:,0,mask].cpu().detach().numpy(), centres[:,1,mask].cpu().detach().numpy(), centres[:,2,mask].cpu().detach().numpy(), force_x.cpu().detach().numpy()* scale, force_y.cpu().detach().numpy()* scale, force_z.cpu().detach().numpy()* scale) #arrow_length_ratio = 0.02
    # ax.quiver(centres[:,0,mask].cpu().detach().numpy(), centres[:,1,mask].cpu().detach().numpy(), centres[:,2,mask].cpu().detach().numpy(), norms.real[:,0,:].cpu().detach().numpy(), norms.real[:,1,:].cpu().detach().numpy(), norms.real[:,2,:].cpu().detach().numpy(),color='red')
    force_quiver_3d(centres[:,:,mask], force_x, force_y, force_z, scale=200)
    # plt.show()


    steps = 60

    print_lines=False
    
    startX = torch.tensor([[-1*diff],[0],[0]])
    endX = torch.tensor([[diff],[0],[0]])
    Es, Exs, Eys, Ezs = get_E_for_fin_diffs(startX, endX, [ball.clone(),walls], board, steps=steps, use_cache=True, print_lines=print_lines,normal_scale=NORMAL_SCALE)
    FxsX, FysX, FzsX = get_force_mesh_along_axis(startX, endX, x, [ball.clone(),walls], board,mask,steps=steps, use_cache=True, print_lines=print_lines,Hs=Es, Hxs=Exs,Hys=Eys, Hzs=Ezs)


    startY = torch.tensor([[0],[-1*diff],[0]])
    endY = torch.tensor([[0],[diff],[0]])
    Es, Exs, Eys, Ezs = get_E_for_fin_diffs(startY, endY, [ball.clone(),walls], board, steps=steps, use_cache=True, print_lines=print_lines,normal_scale=NORMAL_SCALE)
    FxsY, FysY, FzsY = get_force_mesh_along_axis(startX, endX, x, [ball.clone(),walls], board,mask,steps=steps, use_cache=True, print_lines=print_lines,Hs=Es, Hxs=Exs,Hys=Eys, Hzs=Ezs)
    

    startZ = torch.tensor([[0],[0],[-1*diff]])
    endZ = torch.tensor([[0],[0],[diff]])
    Hs, Hxs, Hys, Hzs = get_E_for_fin_diffs(startZ, endZ, [ball.clone(),walls], board, steps=steps, use_cache=True, print_lines=print_lines,normal_scale=NORMAL_SCALE)
    FxsZ, FysZ, FzsZ = get_force_mesh_along_axis(startX, endX, x, [ball.clone(),walls], board,mask,steps=steps, use_cache=True, print_lines=print_lines,Hs=Es, Hxs=Exs,Hys=Eys, Hzs=Ezs)
    

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
        

        xticklabs = [-1* diff, 0 , diff]
        xticks = [0, steps/2 , steps]
        
        plt.subplot(3,1,i+1)
        plt.plot(Fxs, label="$F_x$")
        plt.plot(Fys, label="$F_y$")
        plt.plot(Fzs, label="$F_z-mg$")
        plt.xlabel("$\Delta$" + labs[i] + " (mm)")
        plt.xticks(xticks, xticklabs)
        plt.ylim(ytickmin,ytickmax)
        plt.ylabel("Force (N)")
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.show()