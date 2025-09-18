if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS, propagate_phase
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data
    from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase
    from acoustools.Constants import wavelength

    import torch

    
    board = TRANSDUCERS

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22

    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths)
    centre_scatterer(scatterer)
    print(scatterer.bounds())
    # d = wavelength*2 * 1.05
    d = wavelength*3.56654
    # d = wavelength+0.001
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)


    p = create_points(1,1, y=0,x=0,z=0)
    p2 = create_points(1,1,0,0,-0.002)

    H_method = 'OLS'
    E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True)



    x = iterative_backpropagation(p)
    x = add_lev_sig(x)
    x =translate_hologram(x, dz=0.001)

    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)
    
    def GH_prop(activations, points, scatterer, board, path,use_cache_H, p_ref, H=H ):
        E,F,G,H = compute_E(scatterer, points,board=board, path=path, use_cache_H=use_cache_H, p_ref=p_ref, return_components=True, H=H)


        pressures =  torch.abs(G@H@activations)
        return pressures

    def GH_prop_phase(activations, points, scatterer, board, path,use_cache_H, p_ref, H=H ):
        E,F,G,H = compute_E(scatterer, points,board=board, path=path, use_cache_H=use_cache_H, p_ref=p_ref, return_components=True, H=H)


        pressures =  torch.angle(G@H@activations)
        return pressures



    Visualise(*ABC(0.03), x, points=p2,colour_functions=[propagate_BEM_pressure, GH_prop, propagate_abs, propagate_BEM_phase, GH_prop_phase, propagate_phase], 
              res=(150,150), arangement=(2,3), cmaps=['hot','hot','hot', 'hsv', 'hsv', 'hsv'], link_ax=[0,1,2],
              colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H }, 
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H  }, 
                                    {'board':board, "p_ref":p_ref},
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H }, 
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H  }, 
                                    {'board':board, "p_ref":p_ref}])
