if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, get_rows_in
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, merge_scatterers, translate, insert_parasite, get_centres_as_points
    from acoustools.BEM import propagate_BEM_pressure, compute_E
    from acoustools.Constants import wavelength,k

    import torch

    
    board = TOP_BOARD

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 20 * 0.17

    sphere_path = path+"/Sphere-lam2.stl"
    scatterer = load_scatterer(sphere_path)
    centre_scatterer(scatterer)
    print(scatterer.bounds())
    d = wavelength*3.4
    # d = 0.02341
    # d = wavelength+0.001
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)

    # internal_sphere = load_scatterer(sphere_path)
    # centre_scatterer(internal_sphere)
    # translate(internal_sphere, dx=0.001, dy=0.001, dz=0.001)
    # internal_d = d/8
    # # d = 0.02341
    # # d = wavelength+0.001
    # scale_to_diameter(internal_sphere,internal_d)

    # parasite_scatterer = merge_scatterers(scatterer, internal_sphere)
    delta = 0
    parasite_scatterer = insert_parasite(scatterer, parasite_size=d*0.37, parasite_offset=torch.Tensor([[delta,delta,delta]]))
    sphere_c = get_centres_as_points(scatterer)
    all_c = get_centres_as_points(parasite_scatterer)
    parasite_mask = get_rows_in(all_c, sphere_c, expand=False).logical_not()
    
    alphas = parasite_mask * 0.9
    alphas[parasite_mask.logical_not()] = 1.0 
    
    
    betas = torch.zeros_like(alphas)
    betas[parasite_mask] = 10

    print(alphas, alphas.sum())
    print(betas)

    p = create_points(1,1, y=0,x=0,z=0)

    E = compute_E(parasite_scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref)

    x = iterative_backpropagation(p,A=E)

    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)



    Visualise(*ABC(0.03), x, points=p,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure], res=(150,150),
              colour_function_args=[{'scatterer':parasite_scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref },
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref },
                                    {'scatterer':parasite_scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,"betas":0, "alphas":alphas }], vmax=500)


