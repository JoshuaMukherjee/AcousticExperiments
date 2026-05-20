if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, create_board, transducers, propagate_abs, TOP_BOARD, DTYPE
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centres_as_points
    from acoustools.BEM import propagate_BEM_pressure, compute_E
    from acoustools.Constants import wavelength,k


    
    board = create_board(65, z=0.12)
    board = board.to(DTYPE)
    # board = TOP_BOARD

    path = "../BEMMedia"

    paths = [path+"/Sphere-lam4.stl"]
    scatterer = load_multiple_scatterers(paths)
    centre_scatterer(scatterer)
    d = wavelength*2
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)

    p = get_centres_as_points(scatterer)
    internal_points  = get_CHIEF_points(scatterer, P = 30, start='centre', method='uniform', scale = 0.2, scale_mode='diameter-scale')

    E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False,H_method='LU', return_components=True, internal_points=internal_points)

    x = iterative_backpropagation(p, board=board)

    Visualise(*ABC(0.02, plane='xy'), x,colour_functions=[propagate_BEM_pressure, propagate_abs], res=(300,300),
              colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,
                                     "use_cache_H":False,'k':k,"H":H,'internal_points':internal_points },
                                     {'board':board}], 
              vmax=8000)
