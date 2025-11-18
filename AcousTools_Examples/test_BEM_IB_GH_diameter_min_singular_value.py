if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_CHIEF_points
    from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase, compute_H
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

    H_method = 'OLS'

    pressures = []
    phases = []
    diameters = []
    circumferences = []
    H_phases = []
    sing_valeus = []

    pressures_CHIEF = []
    sing_values_CHIEF = []

    pressures_CHIEF_rect = []
    sing_values_CHIEF_rect = []


    N = 100
    ds = torch.linspace(wavelength*1,3*wavelength, N)

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
        # get_edge_data(scatterer)

        H,A,_ = compute_H(scatterer, board, use_LU=False, use_OLS=True, p_ref=p_ref, return_components=True)
        E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, H=H)
  
        ss = torch.linalg.svdvals(A)
        print(ss, ss.shape, A.shape)
        ss = ss[0]
        sing_val_max = ss[0]
        sing_val_min = ss[-1]
        sing_valeus.append(sing_val_min.item())

        pressure = propagate_BEM_pressure(x, p2, scatterer, board=board, H=H, path=path, p_ref=p_ref)
        pressures.append(pressure.item())



        internal_points  = get_CHIEF_points(scatterer, P = 30, start='centre', method='uniform', scale = 0.1, scale_mode='diameter-scale')
        
        Hc,Ac,_ = compute_H(scatterer, board, use_LU=False, use_OLS=True, p_ref=p_ref, return_components=True, internal_points=internal_points)
        Ec,Fc,Gc,Hc = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, H=Hc, internal_points=internal_points)

        ssc = torch.linalg.svdvals(Ac)[0]
        sing_val_maxc = ssc[0]
        sing_val_minc = ssc[-1]
        print(ssc, ssc.shape, Ac.shape)
        sing_values_CHIEF.append(sing_val_minc.item())

        pressure = propagate_BEM_pressure(x, p2, scatterer, board=board, H=Hc, path=path, p_ref=p_ref)
        pressures_CHIEF.append(pressure.item())


        Hcr,Acr,_ = compute_H(scatterer, board, use_LU=False, use_OLS=True, p_ref=p_ref, return_components=True, internal_points=internal_points, CHIEF_mode='rect')
        Ecr,Fcr,Gcr,Hcr = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, H=Hcr, internal_points=internal_points)


        ssr= torch.linalg.svdvals(Acr)[0]

        sing_val_maxr = ssr[0]
        sing_val_minr = ssr[-1]
        print(ssr, ssr.shape, Acr.shape)
        sing_values_CHIEF_rect.append(sing_val_minr.item())

        pressurer = propagate_BEM_pressure(x, p2, scatterer, board=board, H=Hcr, path=path, p_ref=p_ref)
        pressures_CHIEF_rect.append(pressurer.item())


        diameters.append(d.item())

        print()
        

radii = [d/2 for d in diameters]
kr = [k*r for r in radii]


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.plot(kr, pressures, color='blue')
ax1.plot(kr, pressures_CHIEF, color='blue', linestyle='--')
ax1.plot(kr, pressures_CHIEF_rect, color='blue', linestyle='-.')

ax1.set_ylabel(f"Pressure @ (0,0,-{p2[:,2].item()}m) (Pa)", color='blue')
# plt.xlabel("Sphere Radius ($\lambda$)")
ax1.set_xlabel("kr")


ax2 = ax1.twinx()
ax2.plot(kr, sing_valeus, color='red')
ax2.plot(kr, sing_values_CHIEF, color='red', linestyle='--')
ax2.plot(kr, sing_values_CHIEF_rect, color='red', linestyle='-.')
ax2.set_ylabel("Min singular value of A", color='red')
ax2.set_yscale('log')

plt.show()