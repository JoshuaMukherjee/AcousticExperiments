from acoustools.Utilities import create_points, TRANSDUCERS, propagate_abs, add_lev_sig, transducers
from acoustools.Force import compute_force, force_fin_diff
from acoustools.Gorkov import gorkov_analytical, gorkov_fin_diff
from acoustools.Solvers import wgs
import acoustools.Constants as c
from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points
from acoustools.BEM import BEM_compute_force, BEM_gorkov_analytical, compute_E, get_cache_or_compute_H



from acoustools.Visualiser import ABC, Visualise


board = transducers(16)


path = "../BEMMedia"
sphere_pth =  path+"/Sphere-lam2.stl"
sphere = load_scatterer(sphere_pth, dy=-0.06, dz=-0.04) #Make mesh at 0,0,0
scale_to_diameter(sphere, 0.03)
centres = get_centres_as_points(sphere)

H = get_cache_or_compute_H(sphere, board, path=path)




p = create_points(1,1,0,0,0.015)
x = wgs(p, board=board)
x = add_lev_sig(x, mode='Twin')

stepsize = c.wavelength/64

INDEX = 2

def force(activations, points, board=board):
    Fz = BEM_compute_force(activations, points, board, scatterer=sphere, H=H)[INDEX].unsqueeze(0)
    return Fz

def force_fd(activations, points, board=board):
    Fz = force_fin_diff(activations, points, board=board, stepsize=stepsize, U_function=BEM_gorkov_analytical,
                        U_fun_args={"path":path,"scatterer":sphere,"H":H})[:,INDEX]
    return Fz


def diff_percent(activations, points, board = board):
    f1 = force_fin_diff(activations, points, board=board, stepsize=stepsize, U_function=BEM_gorkov_analytical,
                        U_fun_args={"path":path,"scatterer":sphere,"H":H})[:,INDEX]
    f2 = BEM_compute_force(activations, points, board, scatterer=sphere, H=H)[INDEX].unsqueeze(0)
    return ((f2 - f1) / f1) * 100

R = 40
Visualise(*ABC(0.005, origin=p),x, points=p, colour_functions= [force, force_fd], res=(R,R), link_ax=[0,1])