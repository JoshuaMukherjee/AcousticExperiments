if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
    from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase
    from acoustools.Constants import wavelength,k

    import torch

    
    board = TRANSDUCERS

    path = "../BEMMedia"
    paths = [path+"/Sphere-lam2.stl"]
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22


    p = create_points(1,1, y=0,x=0,z=0)
    p2 = create_points(1,1,0,0,-0.002)

    H_method = 'LU'

    pressures = []
    pressures_res = []
    phases = []
    diameters = []
    circumferences = []
    H_phases = []

    N = 100
    ds = torch.linspace(wavelength*1,2*wavelength, N)


    x = iterative_backpropagation(p)
    x = add_lev_sig(x)
    x =translate_hologram(x, dz=0.001)

    # Visualise(*ABC(0.02),x, res=(200,200), colour_functions=[propagate_abs], colour_function_args=[{'board':board,'p_ref':p_ref}])
    # exit()

    ID = 243


    
    for i,d in enumerate(ds):
        print(i, d, end='\r')

        scatterer = load_multiple_scatterers(paths)
        centre_scatterer(scatterer)
        # d = wavelength*2 * 1.05

        # d = wavelength+0.001
        scale_to_diameter(scatterer,d)

        E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True)


        pressure = propagate_BEM_pressure(x, p2, scatterer, board=board, H=H, path=path, p_ref=p_ref)
        pressures_res.append(pressure.item())


        internal_points  = get_CHIEF_points(scatterer, P = 20, start='centre', method='uniform')
        # print(internal_points.shape)
        
        # internal_points = get_centre_of_mass_as_points(scatterer)
        # internal_points = None
        # get_edge_data(scatterer)

        E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True,internal_points=internal_points)


        pressure = propagate_BEM_pressure(x, p2, scatterer, board=board, H=H, path=path, p_ref=p_ref, internal_points=internal_points)
        pressures.append(pressure.item())

        # phase = propagate_BEM_phase(x, p2, scatterer, board=board, H=H, path=path, p_ref=p_ref, internal_points=internal_points)
        # phases.append((phase).item())

        # H_phase = torch.angle(H@x)[:,ID,:].item()
        # H_phases.append(H_phase)

        diameters.append(d.item())
        circumferences.append((3.1415*d.item())/ wavelength)

        # exit()

radii = [d/2 for d in diameters]
kr = [k*r for r in radii]


import matplotlib.pyplot as plt

# plt.subplot(3,1,1)
plt.plot(radii, pressures_res, label='Standard BEM')
plt.plot(radii, pressures, label='BEM + CHIEF')
plt.ylabel(f"Pressure @ (0,0,-{p2[:,2].item()}m) (Pa)")
plt.xlabel("Sphere Radius (m)")
plt.legend()

plt.show()