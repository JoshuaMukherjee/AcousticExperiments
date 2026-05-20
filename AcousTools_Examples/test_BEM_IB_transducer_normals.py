if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, mesh_to_board
    from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_pressure_grad
    from acoustools.Constants import wavelength,k, P_ref

    import torch

    
    board , norms= mesh_to_board('../BEMMedia/Sphere-lam2.stl')

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22

    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths)
    centre_scatterer(scatterer)
    print(scatterer.bounds())
    d = wavelength*2*0.71
    # d = 0.0673426732
    # d = wavelength * 2
    # d = wavelength * 1.2345
    # d = 0.0234ß
    # d = wavelength+0.004/
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)


    p = create_points(1,1, y=0,x=0,z=0)

    H_method = 'OLS'
    E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True, norms=norms)

    x = iterative_backpropagation(p, board=board, norms=norms)

    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)




    Visualise(*ABC(0.03), x, points=p,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure_grad], res=(100,100),
              colour_function_args=[{'scatterer':scatterer,'board':board, 'transducer_norms':norms,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H },
                                    {'scatterer':scatterer,'board':board, 'transducer_norms':norms,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H, 'norm':True }],   
            link_ax=None)
