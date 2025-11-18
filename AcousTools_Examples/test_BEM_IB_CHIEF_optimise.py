if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points
    from acoustools.BEM import propagate_BEM_pressure, compute_E, find_optimal_CHIEF_points
    from acoustools.Constants import wavelength,k, P_ref

    import torch

    
    board = TOP_BOARD

    path = "../BEMMedia/"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22

    # scatterer_path = 'disc_18mm_lam4.stl'
    scatterer_path = 'Sphere-lam2.stl'
    
    scatterer = load_scatterer(path + scatterer_path)
    centre_scatterer(scatterer)
    print(scatterer.bounds())
    d = wavelength*4
 
    # d = wavelength * 2
    # d = wavelength * 1.2345
    # d = 0.0234ß
    # d = wavelength+0.001
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)


    p = create_points(1,1, y=0,x=0,z=0)

    # internal_points  = get_CHIEF_points(scatterer, P = 20, start='centre', method='uniform')
    internal_points  = get_CHIEF_points(scatterer, P = 30, start='centre', method='uniform', scale = 0.2, scale_mode='diameter-scale')
    Eint,Fint,Gint,Hint = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True, internal_points=internal_points)


    # print(internal_points.shape)
    optimal_points = find_optimal_CHIEF_points(scatterer, board, log=True)

    print(internal_points, optimal_points)

    # print(torch.norm(optimal_points, p=2, dim=1))

    E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True, internal_points=optimal_points)
    Er,Fr,Gr,Hr = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True, internal_points=optimal_points, CHIEF_mode='rect')

    x = iterative_backpropagation(p, board=board)
    x =translate_hologram(x, dz=0.001, board=board)

    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)




    Visualise(*ABC(0.03, plane='xz'), x, points=optimal_points,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure], res=(100,100),
              colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H,"internal_points":optimal_points },
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hint,"internal_points":internal_points },
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k, "H":Hr, 'internal_points':optimal_points},
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k},], vmax=8000)
