from acoustools.Mesh import load_scatterer, get_centres_as_points, scale_to_diameter, get_edge_data, translate, get_normals_as_points, get_areas, load_multiple_scatterers, scatterer_file_name, merge_scatterers
from acoustools.Visualiser import Visualise_mesh,  Visualise, ABC
from acoustools.BEM import get_cache_or_compute_H, grad_H, get_cache_or_compute_H_gradients, propagate_BEM_pressure
from acoustools.Utilities import TOP_BOARD, device, get_rows_in, BOARD_POSITIONS
from acoustools.Solvers import gradient_descent_solver
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Force import force_mesh


import vedo, torch
import pickle

plane = load_scatterer('Media/flat-lam2.stl')
scale_to_diameter(plane, 0.05)
get_edge_data(plane)



centres = get_centres_as_points(plane)


THRESHOLD = 0.024
distances = torch.sqrt(torch.sum(centres**2,dim=1))
idx = (distances.real>THRESHOLD).nonzero()[:,1].flatten()
    
disk = plane.delete_cells(idx.cpu().numpy()).clean()
#2.3cm from base
z_pos = -1*BOARD_POSITIONS + (23/1000)
translate(disk, dz=z_pos)

wall_paths = ["Media/flat-lam2.stl","Media/flat-lam2.stl"]
walls = load_multiple_scatterers(wall_paths,dxs=[-0.198/2,0.198/2],rotys=[90,-90]) #Make mesh at 0,0,0
walls.scale((1,19.3/12,22.5/12),reset=True,origin =False)
# print(walls)
walls.filename = scatterer_file_name(walls)
# print(walls)
get_edge_data(walls)

combined = merge_scatterers(disk, walls)

scatterer_cells = get_centres_as_points(combined)
disk_cells = get_centres_as_points(disk)

mask = get_rows_in(scatterer_cells,disk_cells, expand=False)


board = TOP_BOARD

H = get_cache_or_compute_H(combined, board)
Hx, Hy, Hz = get_cache_or_compute_H_gradients(combined, board,print_lines=True)

norms = get_normals_as_points(combined)
areas = get_areas(combined)
centres = get_centres_as_points(combined)


EPOCHS = 100
BASE_LR = 1
MAX_LR = 10

scheduler = torch.optim.lr_scheduler.CyclicLR
scheduler_args = {
        "max_lr":MAX_LR,
        "base_lr":BASE_LR,
        "cycle_momentum":False,
        "step_size_up":25
    }
grad_params = {
    'scatterer':combined
}

def force_objective(transducer_phases, points, board, targets=None, **objective_params):
    force = force_mesh(transducer_phases,points,norms,areas,board,grad_H,grad_params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    force_masked = force[:,:,mask.squeeze()]
    obj = torch.sum(force_masked[:,2,:]).unsqueeze_(0)
    return (targets - obj)**2


holos = []

for i in range(20):
    TARGET = -0.01 + 0.0005*i

    x = gradient_descent_solver(centres, force_objective,constrains=constrain_phase_only,log=False,\
                                        iters=EPOCHS,lr=BASE_LR, board=board,scheduler=scheduler, scheduler_args=scheduler_args,
                                        targets=torch.tensor([TARGET]).to(device) )

# Visualise_mesh(disk, torch.abs(H@x))


    force = force_mesh(x,centres,norms,areas,board,grad_H,grad_params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    f_sum = torch.sum(force[:,:,mask.squeeze()],dim=2)[:,2].item()
    print(TARGET, f_sum, f_sum*101.97162 )

    holos.append(x)

pickle.dump(holos,open('Media/SavedResults/force_tests.pth','wb'))



# Visualise_mesh(disk, force[:,2,mask.squeeze()])


# A,B,C = ABC(0.12)
# res = (200,200)
# Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':combined,'H':H,'board':board}])