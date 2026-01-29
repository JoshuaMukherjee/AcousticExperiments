if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data
    from acoustools.BEM import propagate_BEM_pressure, compute_E
    from acoustools.Constants import wavelength,k, P_ref

    import torch

    
    board = TOP_BOARD

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22

    paths = [path+"/Sphere-lam2.stl"]



    p = create_points(1,1, y=0,x=0,z=0)
    p2 = create_points(1,1,0,0,-0.002)

    N = 100
    ds = torch.linspace(0.0001, wavelength* 4, steps=N)

    x = iterative_backpropagation(p, board=board)
    x =translate_hologram(x, dz=0.001, board=board)


    pressures = []
    FGHs = []

    Fs= []
    Gs = []
    Hs = []

    GHs = []

    diams = []


    for i,d in enumerate(ds):
        print(i, end='\r')
        
        scatterer = load_multiple_scatterers(paths)
        centre_scatterer(scatterer)
        scale_to_diameter(scatterer,d)

        H_method = 'OLS'
        E,F,G,H = compute_E(scatterer, p2,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True)

        pressure = torch.abs(E@x)
        
        
        FGHs.append(torch.abs(torch.sum(F + G@H)).item())
        
        Fs.append(torch.angle(torch.sum(F)))
        Gs.append(torch.angle(torch.sum(G)))
        Hs.append(torch.angle(torch.sum(H)))

        GHs.append(torch.angle(torch.sum(G@H)))


        diams.append(d.item())
        pressures.append(pressure.item())
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

# ax1.plot(diams, FGHs)
# ax1.plot(diams, Gs, color='r')
# ax1.plot(diams, Gs, color='g')
# ax1.plot(diams, Hs, color='b')
ax1.plot(diams, Hs)
ax1.set_ylabel("(F+GH)")

ax2 = ax1.twinx()
ax2.plot(diams,pressures, color='orange')
ax2.set_ylabel("Pressure (Pa)")

plt.show()