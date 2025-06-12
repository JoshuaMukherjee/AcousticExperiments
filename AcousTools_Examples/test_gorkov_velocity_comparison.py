from acoustools.Gorkov import gorkov_analytical, gorkov_fin_diff, get_gorkov_constants
from acoustools.Visualiser import Visualise, ABC
from acoustools.Utilities import TRANSDUCERS, create_points, propagate_abs, add_lev_sig, forward_model_batched, forward_model_grad, propagate
from acoustools.Solvers import kd_solver
import acoustools.Constants as c

from torch import Tensor
import torch

def gorkov_analytical_velocity(activations: Tensor, points: Tensor,board:Tensor|None=None, axis:str="XYZ", V:float=c.V, **params) -> Tensor:

    if board is None:
        board = TRANSDUCERS

    Fx, Fy, Fz = forward_model_grad(points, transducers=board)
    F = forward_model_batched(points,board)

    p = propagate(activations,points,board,A=F)
    pressure_square = torch.abs(p)**2

    px = (Fx@activations).squeeze(2).unsqueeze(0)
    py = (Fy@activations).squeeze(2).unsqueeze(0)
    pz = (Fz@activations).squeeze(2).unsqueeze(0)

    grad  = torch.cat((px,py,pz),dim=1)

    K1, K2 = get_gorkov_constants(V=V)
    # g = K2*(torch.sum(torch.abs(grad)**2, dim=1))
    p_term = K1* pressure_square

    velocity = grad /( 1j * c.p_0 * c.angular_frequency)
    velocity_time_average = 1/2 * torch.sum(velocity * velocity.conj().resolve_conj(), dim=1, keepdim=True).real

    f1 = 1 -( (c.p_0*c.c_0**2)/(c.p_p * c.c_p**2))
    f2 = (2 * (c.p_p - c.p_0))/(2*c.p_p + c.p_0)

    # print((2 * c.pi* c.radius**3 * f2*c.p_0/2 * velocity_time_average)/ g, sep='\n')
    # exit()
    #  This is too big by factor of 91.1250

    U = 2 * c.pi* c.R**3 * ((f1/(3*c.p_0*c.c_0**2) * 1/2*pressure_square) - (f2*c.p_0/2 * velocity_time_average) )
    return U.permute(0,2,1)

def gorkov_ratio(activations: Tensor, points: Tensor,board:Tensor|None=None, axis:str="XYZ", V:float=c.V, **params) -> Tensor:
    U = gorkov_analytical(activations, points, board)
    U_v = gorkov_analytical_velocity(activations, points, board)
    return U / U_v



board = TRANSDUCERS
p = create_points(1,1)
x = kd_solver(p, board=board)
x = add_lev_sig(x)

r = 100
Visualise(*ABC(0.02, origin=p), x, points=p, 
          colour_functions=[propagate_abs, gorkov_analytical, gorkov_analytical_velocity, gorkov_ratio], 
          colour_function_args=[{'board':board}, {'board':board}, {'board':board},{'board':board}],
          link_ax=None, res = (r,r))
