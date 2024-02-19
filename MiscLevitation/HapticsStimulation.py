from acoustools.Utilities import propagate, create_points, device, TRANSDUCERS
from acoustools.Solvers import gradient_descent_solver
from acoustools.Visualiser import Visualise
from acoustools.Optimise.Constraints import constrain_phase_only
import torch

A = 0.124
B = 0.000326
C = 0.00692
D = -0.236
E = -0.0000000133
F = -0.0000000413
G = -0.0000471
H = -0.000653
I = -0.0000892
J = 0.0591

MPT = 750.58


def compute_PI(activations, points,board, fd, Nf):

    p = propagate(activations,points,board)
    pressure = torch.abs(p)
    p_mpt = pressure - MPT

    return A + B * p_mpt + C * fd + D* Nf + E*p_mpt**2 + F*p_mpt*fd + G*p_mpt*Nf + H*fd**2 + I*fd*Nf + J*Nf**2

def objective(transducer_phases, points, board, targets, **objective_params):

    fd = objective_params['fd']
    Nf = objective_params['Nf']
    PI = compute_PI(transducer_phases, points,board, fd, Nf)
    return torch.sum((PI-targets)**2).unsqueeze(0)


if __name__ == "__main__":
    points = torch.tensor([[0.03, 0.03, -0.03, -0.03], [0.03, -0.03, 0.03, -0.03], [ 0,0,0,0]]).to(device).unsqueeze(0)
    # points = torch.tensor([[0.06, 0.06, -0.06, -0.06], [0.06, -0.06, 0.06, -0.06], [ 0,0,0,0]]).to(device).unsqueeze(0)

    fd = 20
    Nf = 4
    params = {
        'Nf':Nf,
        'fd':fd
    }

    # targets= torch.tensor([0.8,0.7,0.9,0.35]).to(device) #Seems to only work for PI = [0,0.35]
    targets= torch.tensor([0.3,0.2,0.25,0.35]).to(device) #Seems to only work for PI = [0,0.35]

    lr = 1
    Epochs=200
    x =  gradient_descent_solver(points, objective, targets=targets, objective_params=params,log =True,iters=Epochs, lr=lr, constrains=constrain_phase_only)
    pressure = torch.abs(propagate(x,points))
    pressure_perc = pressure - MPT
    print(pressure)
    print(compute_PI(x, points,TRANSDUCERS, fd, Nf))
    # print(torch.mean(compute_PI(x, points,TRANSDUCERS, fd, Nf)), torch.mean(targets))
    print(targets)
    # print(pressure_perc/torch.max(pressure_perc))


    A = torch.tensor((-0.09, 0.09,0))
    B = torch.tensor((0.09, 0.09,0))
    C = torch.tensor((-0.09, -0.09,0))
    normal = (0,1,0)
    origin = (0,0,0)

    # Visualise(A,B,C, x, points=points)

