from acoustools.Utilities import create_points, BOTTOM_BOARD
from acoustools.Mesh import load_scatterer, calculate_features, get_edge_data, get_normals_as_points, get_centres_as_points, centre_scatterer
from acoustools.BEM import compute_E, get_cache_or_compute_H, propagate_BEM_pressure
from acoustools.Visualiser import Visualise, ABC, force_quiver_3d, Visualise_mesh
from acoustools.Solvers import wgs

import torch, vedo

path = '../BEMMedia'


with torch.no_grad():

    board = BOTTOM_BOARD

    hand = load_scatterer('/Hand-0-lam2.STL', root_path=path, roty=180).fill_holes(10)
    # hand=load_scatterer('/sphere-lam2.stl',root_path=path)
    centre_scatterer(hand)
    hand.fill_holes(10)

    hand.subdivide(2)
    hand.clean()
    print(hand.is_closed())


    normals = get_normals_as_points(hand)
    centres = get_centres_as_points(hand)

    # force_quiver_3d(centres, normals[:,0,:].real, normals[:,1,:].real, normals[:,2,:].real, scale=0.01)

    calculate_features(hand)
    get_edge_data(hand)

    vedo.save(hand,path+'/hand-0-lam4.stl', False)
    exit()

    # print(hand)
    p = create_points(1,1,0,0,-0.07)
    # vedo.show(hand, axes=1)

    E,F,G,H = compute_E(hand, p, board,return_components=True, path=path)

    x = wgs(p, board=board, A=E, iter=500)

    A,B,C = ABC(0.15, 'xz')

    Visualise(A,B,C,x,p, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':hand,"H":H,"board":board}], res=(150,150))





