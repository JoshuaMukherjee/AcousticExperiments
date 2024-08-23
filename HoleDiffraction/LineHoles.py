from acoustools.Mesh import load_scatterer, calculate_features, get_normals_as_points, get_centres_as_points, scatterer_file_name, merge_scatterers
from acoustools.BEM import compute_E, propagate_BEM_pressure, get_cache_or_compute_H
from acoustools.Utilities import create_points, BOTTOM_BOARD, device
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise_single, ABC, force_quiver
from acoustools.Constants import wavelength

import vedo, torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation


path = '../BEMMedia'

board = BOTTOM_BOARD

THRESHOLD = wavelength*1

sizes = [5,4.5,4,3.5,3,2.5,2,1.5,1,0.5,0.25,0.125]

with torch.no_grad():
    def next_image(i):
        scatterer = load_scatterer('/flat-lam2.stl', root_path=path, dz=0, roty=180)

        norms = get_normals_as_points(scatterer)
        centres = get_centres_as_points(scatterer)
        THRESHOLD = wavelength*sizes[i]
        idx = (torch.abs(centres[:,0,:])<THRESHOLD).nonzero()[:,1].flatten()
        # 

        scatterer = scatterer.delete_cells(idx.cpu().numpy()).clean()
        # vedo.show(scatterer)
        
        scatterer.filename = scatterer_file_name(scatterer)
        calculate_features(scatterer)


        p = create_points(1,1,0,0,0.03)

        E,F,G,H = compute_E(scatterer, p, board, return_components=True, path=path)

        x = wgs(p, board=board, A=E)

        A,B,C = ABC(0.05)

        img_ax.clear()
        im = Visualise_single(A,B,C, x, colour_function=propagate_BEM_pressure, colour_function_args={"board":board, "H":H, "scatterer":scatterer}, res=(200,200))


        img_ax.matshow(im.cpu().detach(),cmap='hot') 

        print(i)
        scatterer = None
        x=None
        torch.cuda.empty_cache()


    fig = plt.figure()
    img_ax = fig.add_subplot(1,1,1)

    lap_animation = animation.FuncAnimation(fig, next_image, frames=range(len(sizes)), interval=1000)

    # lap_animation = animation.FuncAnimation(fig, traverse, frames=[1,2,3,4], interval=THRESHOLD*1000)
    lap_animation.save('Results.gif', dpi=80, writer='imagemagick')
