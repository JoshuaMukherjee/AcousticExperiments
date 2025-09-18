if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data
    from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase
    from acoustools.Constants import wavelength

    import torch

    
    board = TRANSDUCERS





    path = "../BEMMedia"
    paths = [path+"/Sphere-lam1.stl", path+"/Sphere-lam2.stl", path+"/Sphere-lam4.stl"]
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22


    p = create_points(1,1, y=0,x=0,z=0)
    p2 = create_points(1,1,0,0,-0.002)

    H_method = 'OLS'

    all_pressure = []
    all_phases = []
    all_ds = []

    N = 100
    ds = torch.linspace(wavelength*1,2*wavelength, N)


    x = iterative_backpropagation(p)
    x = add_lev_sig(x)
    x =translate_hologram(x, dz=0.001)


    for pth in paths:

        pressures = []
        phases = []
        diameters = []


        for i,d in enumerate(ds):
            print(i, d, end='\r')

            scatterer = load_scatterer(pth)
            centre_scatterer(scatterer)
            # d = wavelength*2 * 1.05

            # d = wavelength+0.001
            scale_to_diameter(scatterer,d)
            # get_edge_data(scatterer)

            E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True)


            pressure = propagate_BEM_pressure(x, p2, scatterer, board=board, H=H, path=path, p_ref=p_ref)
            pressures.append(pressure.item())

            phase = propagate_BEM_phase(x, p2, scatterer, board=board, H=H, path=path, p_ref=p_ref)
            phases.append((phase).item())

            diameters.append(d.item()/ wavelength)


        all_pressure.append(pressures)
        all_phases.append(phases)
        all_ds.append(diameters)


import matplotlib.pyplot as plt

for i in range(len(all_pressure)):

    pressures = all_pressure[i]
    phases = all_phases[i]
    diameters = all_ds[i]


    radii = [d/2 for d in diameters]


    plt.subplot(2,1,1)
    plt.plot(radii, pressures, label=paths[i])
    plt.ylabel(f"Pressure @ (0,0,-{p2[:,2].item()}m) (Pa)")
    plt.xlabel("Sphere Radius ($\lambda$)")

    plt.subplot(2,1,2)

    plt.plot(radii, phases, label=paths[i])
    plt.ylabel(f"Phase @ (0,0,-{p2[:,2].item()}m) (rad)")
    plt.xlabel("Sphere Radius ($\lambda$)")
    
    
plt.legend()
plt.show()