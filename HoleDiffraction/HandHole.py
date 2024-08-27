from acoustools.Utilities import create_points, BOTTOM_BOARD
from acoustools.Mesh import load_scatterer, calculate_features, get_edge_data, get_normals_as_points, get_centres_as_points, centre_scatterer, get_lines_from_plane, scatterer_file_name
from acoustools.BEM import compute_E, get_cache_or_compute_H, propagate_BEM_pressure
from acoustools.Visualiser import Visualise, ABC, force_quiver_3d, Visualise_mesh
from acoustools.Solvers import wgs

import torch, vedo
import numpy as np

path = '../BEMMedia'


with torch.no_grad():

    board = BOTTOM_BOARD

    hand = load_scatterer('/Hand-0-lam2.STL', root_path=path, roty=180).fill_holes(10)
    # hand = load_scatterer('/Participant_1_29.obj', root_path=path, roty=180)
    # correction = centre_scatterer(hand)
    # hand = hand.clean().smooth()

    # edges = hand.count_vertices()
    # mask = np.where(edges != 3)[0]
    # hand.delete_cells(mask)
    # hand.filename = scatterer_file_name(hand)


    # hand.subdivide(1)        
    # # hand = hand.decimate(1)
    
    # hand.compute_normals()
    # hand = hand.reverse(cells=True, normals=True)

    # hand.compute_cell_size()
    # hand.filename = scatterer_file_name(hand)


#   .fill_holes(10)
    # hand=load_scatterer('/sphere-lam2.stl',root_path=path)
    centre_scatterer(hand)
    # hand.fill_holes(10)

    hand.subdivide(1)
    hand.clean()
    print(hand.is_closed())


    normals = get_normals_as_points(hand)
    centres = get_centres_as_points(hand)

    # force_quiver_3d(centres, normals[:,0,:].real, normals[:,1,:].real, normals[:,2,:].real, scale=0.01)

    print(hand)
    calculate_features(hand)
    get_edge_data(hand)

    # vedo.save(hand,path+'/hand-0-lam4.stl', False)
    # exit()

    # print(hand)
    p = create_points(1,1,0,0,-0.07)
    # vedo.show(hand, axes=1)

    E,F,G,H = compute_E(hand, p, board,return_components=True, path=path)

    x = wgs(p, board=board, A=E, iter=500)

    A,B,C = ABC(0.15, 'xz')


    normal = (0,1,0)
    origin = (0,0,0)
    line_params = {"scatterer":hand,"origin":origin,"normal":normal}
    

    Visualise(A,B,C,x,p, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':hand,"H":H,"board":board}], res=(150,150),
              add_lines_functions=[get_lines_from_plane],add_line_args=[line_params])





