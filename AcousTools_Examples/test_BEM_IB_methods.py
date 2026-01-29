if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS, propagate_phase, DTYPE
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, get_normals_as_points, get_CHIEF_points
    from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase, compute_A, compute_bs
    from acoustools.Constants import wavelength, k

    import torch

    
    board = TRANSDUCERS

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22

    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths)
    centre_scatterer(scatterer)
    # print(scatterer.bounds())
    # d = wavelength*2 * 1.05
    # d = wavelength * 3.34
    # d = wavelength * 2
    # d = 0.02
    d = wavelength*2*0.71
    # d = 0.0001
    # d = wavelength+0.001
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)



    internal_points  = get_CHIEF_points(scatterer, P = 50 , start='centre', method='uniform', scale = 0.2, scale_mode='diameter-scale')
    # print(internal_points)
    # print(internal_points.shape)
    # internal_points = None



    p = create_points(1,1, y=0,x=0,z=0)
    p2 = create_points(1,1,0,0,-0.002)

    h = 1e-3
    alpha = (1j)/(10*k)

    H_method = 'OLS'
    E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True)
    E,F,G,H_CHIEF = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, internal_points=internal_points)
    E,F,G,H_BM= compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, h=h, BM_alpha=alpha)
    E,F,G,H_BM_CHIEF= compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, h=h, BM_alpha=alpha, internal_points=internal_points)

    A = compute_A(scatterer)
    bs = compute_bs(scatterer,board, p_ref=p_ref)

    z = torch.zeros_like(H) + torch.rand_like(H) *1e-3
    Hmin = torch.linalg.inv(A) @ z
    H = H-Hmin*k

    
    print(torch.linalg.inv(A) @ z)

    # exit()

    x = iterative_backpropagation(p)
    x = add_lev_sig(x)
    x =translate_hologram(x, dz=0.001)

    print(propagate_BEM_pressure(x, p2, scatterer, board=board, H=H, path=path, p_ref=p_ref, internal_points=internal_points))



    r = 200
    Visualise(*ABC(d), x,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure],  
            #   points=internal_points,
              res=(r,r), arangement=(2,2), cmaps=['hot','hot','hot', 'hot'], vmax=1000,
              titles=['BEM', 'CHIEF', "BM", "CHIEF-BM"],
              colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H }, 
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H_CHIEF, "internal_points":internal_points  },
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H_BM  },
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "H":H_BM_CHIEF , "internal_points":internal_points }

              ]
    )