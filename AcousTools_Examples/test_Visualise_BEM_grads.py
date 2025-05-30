
from acoustools.Utilities import forward_model_batched, forward_model_grad, forward_model_second_derivative_unmixed, forward_model_second_derivative_mixed, TRANSDUCERS, create_points, add_lev_sig, propagate,DTYPE, transducers
from acoustools.Solvers import wgs
from acoustools.Gorkov import get_finite_diff_points_all_axis
from acoustools.Utilities import device, propagate_abs
import acoustools.Constants as c
from acoustools.Mesh import load_scatterer
from acoustools.BEM import compute_E, BEM_forward_model_grad, BEM_forward_model_second_derivative_mixed, BEM_forward_model_second_derivative_unmixed, propagate_BEM_pressure, get_cache_or_compute_H

from acoustools.Visualiser import Visualise, ABC
import torch

from torch import Tensor

torch.random.manual_seed(1)


board = transducers(1)


path = "../BEMMedia"
sphere_pth =  path+"/Sphere-lam2.stl"
sphere = load_scatterer(sphere_pth, dy=-0.06, dz=-0.04) #Make mesh at 0,0,0

N=1
B=1
D=3
H = get_cache_or_compute_H(sphere, board, path=path)

def propagate_grads_unmixed(activations: Tensor, points: Tensor, scatterer = sphere, board: Tensor | None = board):
    SCALE  = 0.00000001
    Exx, Eyy, Ezz = BEM_forward_model_second_derivative_unmixed(points, sphere, board, H=H, path=path)
    Pxx = torch.abs(Exx@activations)
    Pyy = torch.abs(Eyy@activations)
    Pzz = torch.abs(Ezz@activations)
    mx = torch.max(torch.stack((Pxx, Pyy, Pzz)))

    return torch.stack((Pxx/(SCALE*mx), Pyy/(SCALE*mx), Pzz/(SCALE*mx)), dim=2)

def propagate_grads(activations: Tensor, points: Tensor, scatterer = sphere, board: Tensor | None = board):
    SCALE = 0.1
    Ex, Ey, Ez = BEM_forward_model_grad(points, sphere, board, H=H, path=path)
    Px = torch.abs(Ex@activations)
    Py = torch.abs(Ey@activations)
    Pz = torch.abs(Ez@activations)
    mx = torch.max(torch.stack((Px, Py, Pz)))

    return torch.stack((Px/(SCALE*mx), Py/(SCALE*mx), Pz/(SCALE*mx)), dim=2)




for i in range(1):
    # points = create_points(N,B,x=0.02, y=-0.005, z=-0.04)
    points = create_points(N,B, y=0, z=0.02)
    print(points)
    points = torch.autograd.Variable(points.data, requires_grad=True).to(device).to(DTYPE)

    E,F,G,H = compute_E(sphere, points, board, return_components=True, path=path)


    activations = wgs(points, A=E).to(DTYPE)

    res = 100
    Visualise(*ABC(0.1, origin=points), activations, colour_functions=[propagate_BEM_pressure, propagate_grads_unmixed], 
              colour_function_args=[{"board":board,"path":path,"H":H,"scatterer":sphere},{}], res=(res,res), depth=0)
    exit()
 