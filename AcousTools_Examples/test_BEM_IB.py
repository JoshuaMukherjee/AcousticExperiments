if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data
    from acoustools.BEM import propagate_BEM_pressure, compute_E
    from acoustools.Constants import wavelength

    import torch

    
    def propagate_abs_sum_objective_BEM(transducer_phases, points, board, targets, **objective_params):
        scatterer = objective_params['scatterer']
        E = objective_params['E']
        return torch.sum(propagate_BEM_pressure(transducer_phases,points,scatterer=scatterer,board=board,E=E),dim=1).squeeze(0)

    board = TOP_BOARD

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])


    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths)
    centre_scatterer(scatterer)
    print(scatterer.bounds())
    d = wavelength*2 * 1.05
    # d = wavelength+0.001
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)


    p = create_points(1,1, y=0,x=0,z=0)

    E = compute_E(scatterer, p,board=board, path=path, use_cache_H=True)

    x = iterative_backpropagation(p,A=E)

    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)

    Visualise(*ABC(0.03), x, points=p,vmax=5000,colour_functions=[propagate_BEM_pressure], res=(150,150),
              colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":True }])
