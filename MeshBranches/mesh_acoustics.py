from acoustools.Mesh import load_scatterer, get_edge_data
from acoustools.BEM import compute_E, propagate_BEM_pressure
from acoustools.Utilities import BOTTOM_BOARD, create_points
from acoustools.Visualiser import Visualise_mesh, ABC, Visualise, Visualise_single
from acoustools.Solvers import wgs

import vedo, torch, pickle

import matplotlib.pyplot as plt

board = BOTTOM_BOARD


path = '../BEMMedia'

A,B,C = ABC(0.15, 'xz')

mean_pressures = {}
max_pressures = {}
min_pressures = {}
imgs = {}

with torch.no_grad():

    for set in range(10):
        # mean_pressures[set] = []
        # max_pressures[set] = []
        # min_pressures[set] = []
        imgs[set] = []
        for i in range(30):
            print(set, i)
            scatterer = load_scatterer(f'./MeshBranches/Meshes/icoso/set{set}/m{i}.stl', dz=0.04)
            p = create_points(1,1,0,0,0)

            E,F,G,H = compute_E(scatterer, p, board, path=path, return_components=True)

            x = wgs(p, board=board, A=E)

            im = Visualise_single(A,B,C,x,propagate_BEM_pressure,colour_function_args={'scatterer':scatterer,"H":H,"board":board}, res = (100,100))
            imgs[set].append(im.cpu().detach())

            # mean_p = torch.mean(im)
            # max_p = torch.max(im)
            # min_p = torch.min(im)

            # mean_pressures[set].append(mean_p)
            # max_pressures[set].append(max_p)
            # min_pressures[set].append(min_p)
            
            
            torch.cuda.empty_cache()




    # pickle.dump(mean_pressures, open('mean_pressure.pth','wb'))
    # pickle.dump(max_pressures, open('max_pressure.pth','wb'))
    # pickle.dump(min_pressures, open('min_pressure.pth','wb'))
    pickle.dump(imgs, open('imgs.pth','wb'))