from acoustools.Utilities import propagate, create_points, device
from acoustools.Solvers import gradient_descent_solver
from acoustools.Visualiser import Visualise
from acoustools.Optimise.Constraints import constrain_phase_only
import torch

def compute_PI(activations, points, fd, Nf):
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

    p = propagate(activations,points)
    pressure = torch.abs(p)
    p_mpt = pressure - MPT

    return A + B * p_mpt + C * fd + D* Nf + E*p_mpt**2 + F*p_mpt*fd + G*p_mpt*Nf + H*fd**2 + I*fd*Nf + J*Nf**2

def objective(transducer_phases, points, board, targets, **objective_params):

    fd = objective_params['fd']
    Nf = objective_params['Nf']
    PI = compute_PI(transducer_phases, points, fd, Nf)
    return torch.sum((PI-targets)**2).unsqueeze(0)


if __name__ == "__main__":
    points = create_points(4,1,z=0) #create 4 points in the z=0 plane
    fd = 20
    Nf = 1
    params = {
        'Nf':Nf,
        'fd':fd
    }

    targets= torch.tensor([0.7,0.8,0.65,0.9]).to(device)

    lr = 10
    Epochs=2000
    x =  gradient_descent_solver(points, objective, targets=targets, objective_params=params,log =True,iters=Epochs, lr=lr, constrains=constrain_phase_only)
    print(torch.abs(propagate(x,points)))
    print(compute_PI(x, points, fd, Nf))

    A = torch.tensor((-0.09, 0.09,0))
    B = torch.tensor((0.09, 0.09,0))
    C = torch.tensor((-0.09, -0.09,0))
    normal = (0,1,0)
    origin = (0,0,0)

    Visualise(A,B,C, x, points=points)

