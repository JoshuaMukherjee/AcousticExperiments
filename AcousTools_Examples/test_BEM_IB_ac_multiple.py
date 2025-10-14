if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, get_centre_of_mass_as_points, get_centres_as_points, get_normals_as_points
    from acoustools.BEM import propagate_BEM_pressure, compute_E
    from acoustools.Constants import wavelength,k

    import torch

    
    board = TOP_BOARD

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 20 * 0.17

    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths)
    centre_scatterer(scatterer)
    print(scatterer.bounds())
    d = wavelength*2*0.7175
    # d = 0.0234
    # d = wavelength+0.001
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)

    scatterer_sub = scatterer.clone()
    scatterer_sub = scatterer_sub.decimate(0.005)

    a_s = get_centres_as_points(scatterer_sub) - 0.1*d/2 * get_normals_as_points(scatterer_sub)
    print(a_s.shape)

    # a = get_centre_of_mass_as_points(scatterer)
    c=1

    p = create_points(1,1, y=0,x=0,z=0)

    E = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref)

    x = iterative_backpropagation(p,A=E)

    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)



    # Visualise(*ABC(0.1), x, points=p,colour_functions=[propagate_abs], res=(200,200),
    #           colour_function_args=[{"board":board}], vmax=8000)


    Visualise(*ABC(0.03), x, points=a_s,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure], res=(100,100),
              colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, "a":a_s, "c":c },
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref, }], vmax=8000)
