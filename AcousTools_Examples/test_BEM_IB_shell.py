if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, merge_scatterers, get_centres_as_points, get_normals_as_points, get_CHIEF_points
    from acoustools.BEM import propagate_BEM_pressure, compute_E
    from acoustools.Constants import wavelength,k, P_ref

    import torch

    
    board = TOP_BOARD

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22

    # paths = [path+"/Sphere-lam2.stl"]
    paths = [path+'/Cube-lam4.stl']
    # paths = [path + '/disc_12mm_lam4.stl']

    outer = load_multiple_scatterers(paths)
    centre_scatterer(outer)
    print(outer.bounds())
    # d = wavelength*2*0.71
    d = wavelength * 2.5
    scale_to_diameter(outer,d)
    get_edge_data(outer)

    centers = get_centres_as_points(outer)
    M = centers.shape[2]


    inner = load_multiple_scatterers(paths)
    inner.flip_normals()
    centre_scatterer(inner)
    # print(scatterer.bounds())
    inner_d = d *0.75
    scale_to_diameter(inner,inner_d)

    centers = get_centres_as_points(inner)
    N = centers.shape[2]
    # get_edge_data(inner)


    scatterer = merge_scatterers(outer, inner)


    alphas_out = torch.ones((1,M))
    alphas_in = torch.zeros((1,N)) + 0.1
    alphas = torch.cat((alphas_out, alphas_in), dim=1)

    print(alphas.shape)


    p = create_points(1,1, y=0,x=0,z=0)

    H_method = 'OLS'
    E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True, alphas=alphas)


    internal_points  = get_CHIEF_points(scatterer, P = 30, start='centre', method='uniform', scale = 0.2, scale_mode='diameter-scale')
    E_C, F_C, G_C, H_C = compute_E(outer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True, internal_points=internal_points)

    x = iterative_backpropagation(p, board=board)
    x =translate_hologram(x, dz=0.001, board=board)

    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)




    Visualise(*ABC(0.03), x, points=p,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure,'-'], res=(100,100),
              colour_function_args=[{'scatterer':outer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,},
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H, "alphas":alphas },
                                    {'scatterer':outer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H_C,"internal_points":internal_points},
                                    {'ids':[1,2]}
                                    ], vmax=1000)
