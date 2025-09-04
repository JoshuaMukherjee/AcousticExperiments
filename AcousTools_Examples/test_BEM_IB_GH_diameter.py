if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data
    from acoustools.BEM import propagate_BEM_pressure, compute_E
    from acoustools.Constants import wavelength

    import torch

    
    board = TRANSDUCERS

    path = "../BEMMedia"
    paths = [path+"/Sphere-solidworks-lam2.stl"]
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22


    p = create_points(1,1, y=0,x=0,z=0)
    p2 = create_points(1,1,0,0,-0.002)

    H_method = 'OLS'

    pressures = []
    diameters = []

    N = 200
    ds = torch.linspace(wavelength*0.5, 4*wavelength, N)

    
    for i,d in enumerate(ds):
        print(i, d, end='\r')

        scatterer = load_multiple_scatterers(paths)
        centre_scatterer(scatterer)
        # d = wavelength*2 * 1.05

        # d = wavelength+0.001
        scale_to_diameter(scatterer,d)
        # get_edge_data(scatterer)

        E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True)



        x = iterative_backpropagation(p,A=E)
        x = add_lev_sig(x)
        x =translate_hologram(x, dz=0.001)

        pressure = propagate_BEM_pressure(x, p2, scatterer, board=board, H=H, path=path, p_ref=p_ref)
        pressures.append(pressure.item())

        diameters.append(d.item()/ wavelength)


import matplotlib.pyplot as plt

plt.plot(diameters, pressures)
plt.ylabel("Pressure @ (0,0,-2mm) (Pa)")
plt.xlabel("Sphere Diameter ($\lambda$)")
plt.show()