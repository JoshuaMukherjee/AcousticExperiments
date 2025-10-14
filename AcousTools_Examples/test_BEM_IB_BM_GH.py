if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS, propagate_phase
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data
    from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase
    from acoustools.Constants import wavelength, k, P_ref

    import torch

    
    board = TOP_BOARD

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = P_ref

    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths)
    centre_scatterer(scatterer)
    print(scatterer.bounds())
    # d = wavelength*2 * 1.05
    # d = wavelength * 1.2345
    # d = wavelength * 2
    # d = wavelength+0.001
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)


    p = create_points(1,1, y=0,x=0,z=0)
    p2 = create_points(1,1,0,0,-0.002)

    alpha = 1j/k


    H_method = 'OLS'
    E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, BM_c=alpha)


    x = iterative_backpropagation(p, board=board)

    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)
    
    def GH_prop(activations, points, scatterer, board, path,use_cache_H, p_ref, H=H,BM_c=None ):
        E,F,G,H = compute_E(scatterer, points,board=board, path=path, use_cache_H=use_cache_H, p_ref=p_ref, return_components=True, H=H, BM_c=BM_c)


        pressures =  torch.abs(G@H@activations)
        return pressures

    def GH_prop_phase(activations, points, scatterer, board, path,use_cache_H, p_ref, H=H,BM_c=None ):
        E,F,G,H = compute_E(scatterer, points,board=board, path=path, use_cache_H=use_cache_H, p_ref=p_ref, return_components=True, H=H, BM_c=BM_c)


        phases =  torch.angle(G@H@activations)

        return phases



    Visualise(*ABC(0.03), x,colour_functions=[propagate_BEM_pressure, GH_prop, propagate_abs, propagate_BEM_phase, GH_prop_phase, propagate_phase], 
              res=(150,150), arangement=(2,3), cmaps=['hot','hot','hot', 'hsv', 'hsv', 'hsv'], link_ax=[0,1,2], 
              colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H,"BM_c":alpha }, 
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H,"BM_c":alpha  }, 
                                    {'board':board, "p_ref":p_ref},
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H,"BM_c":alpha }, 
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H,"BM_c":alpha  }, 
                                    {'board':board, "p_ref":p_ref}])
