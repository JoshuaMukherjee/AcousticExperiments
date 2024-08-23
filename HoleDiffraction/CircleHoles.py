from acoustools.Mesh import load_scatterer, calculate_features, get_normals_as_points, get_centres_as_points, scatterer_file_name, get_edge_data
from acoustools.BEM import compute_E, propagate_BEM_pressure, get_cache_or_compute_H
from acoustools.Utilities import create_points, BOTTOM_BOARD, device
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, ABC, force_quiver
from acoustools.Constants import wavelength

import vedo, torch

path = '../BEMMedia'

board = BOTTOM_BOARD

THRESHOLD = wavelength*2

with torch.no_grad():
    scatterer = load_scatterer('/flat-lam4.stl', root_path=path, dz=0, roty=180)

    norms = get_normals_as_points(scatterer)
    centres = get_centres_as_points(scatterer)
    distances = torch.sqrt(torch.sum(centres**2,dim=1))
    idx = (distances.real<THRESHOLD).nonzero()[:,1].flatten()
    

    scatterer = scatterer.delete_cells(idx.cpu().numpy()).clean()
    
    scatterer.filename = scatterer_file_name(scatterer)
    calculate_features(scatterer)

    p = create_points(1,1,0,0,0.03)

    E,F,G,H = compute_E(scatterer, p, board, return_components=True, path=path)

    x = wgs(p, board=board, A=E)

    A,B,C = ABC(0.05)

    Visualise(A,B,C, x, p, colour_functions=[propagate_BEM_pressure], colour_function_args=[{"board":board, "H":H, "scatterer":scatterer}], res=(100,100))
