if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS, propagate_phase
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC, Visualise_single
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, get_centres_as_points, load_scatterer, merge_scatterers, insert_parasite, get_diameter
    from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase
    from acoustools.Constants import wavelength
    from acoustools.Export.CSV import write_to_file

    import torch, vedo, os
    import matplotlib.pyplot as plt


    # import os

    # for f in sorted(os.listdir("AcousTools_Examples/outputs/BEM_IB_GH_animate")):
    #     if '.stl' in f:
    #         scatterer = load_scatterer("AcousTools_Examples/outputs/BEM_IB_GH_animate/"+f)
    #         get_edge_data(scatterer)
    #         print(scatterer.bounds()/wavelength)
    #         print()

    # exit()
    
    board = TRANSDUCERS

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22

    cache_H = False

    paths = [path+"/Sphere-lam1.stl"]


    p = create_points(1,1, y=0,x=0,z=-0.001)
    x = iterative_backpropagation(p)
    x = add_lev_sig(x)



    # x =translate_hologram(x, dz=0.001)

    p2 = create_points(1,1,0,0,-0.002)


    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)
    
    def GH_prop(activations, points, scatterer, board, path,use_cache_H, p_ref, H):
        E,F,G,H = compute_E(scatterer, points,board=board, path=path, use_cache_H=use_cache_H, p_ref=p_ref, return_components=True, H=H)


        pressures =  torch.abs(G@H@activations)
        return pressures

    def GH_prop_phase(activations, points, scatterer, board, path,use_cache_H, p_ref, H):
        E,F,G,H = compute_E(scatterer, points,board=board, path=path, use_cache_H=use_cache_H, p_ref=p_ref, return_components=True, H=H)


        pressures =  torch.angle(G@H@activations)
        return pressures


    N = 20
    # ds = torch.linspace(wavelength*3.6, wavelength*3.8, N)
    # rs_wavelength = [0.5025, 0.7155, 0.9185, 1.001, 1.114, 1.229, 1.447]
    rs_wavelength = [0.6, 1.3]
    rs = [wavelength*r for r in rs_wavelength]
    ds = [r*2 for r in rs]


    
    res = (100,100)

    frames = []

    ID = 743

    # for i,d in enumerate(ds):
    for i,paths in enumerate(os.listdir("/Users/joshuamukherjee/Desktop/Education/University/UCL/PhD/BEMMedia/BEMTest")):
        paths=["/Users/joshuamukherjee/Desktop/Education/University/UCL/PhD/BEMMedia/BEMTest/"+paths,]
        print(i, end='\r')


        frame = []

        scatterer = load_multiple_scatterers(paths)
        d = get_diameter(scatterer)
        scatterer = insert_parasite(scatterer, parasite_size=d*0.7)

        

        
        # centre_scatterer(scatterer
        # scale_to_diameter(scatterer,d)

        # scatterer.write(f"Sphere-lam2-d{d}.stl", binary=False)
        # continue

        # get_edge_data(scatterer)


        centres = get_centres_as_points(scatterer)
        # if i ==0: 
        #     c = centres[:,:,ID]
        #     abc = ABC(0.005, origin=c)

        abc = ABC(0.05)
   

        H_method = 'OLS'
        E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=cache_H, p_ref=p_ref,H_method=H_method, return_components=True)

        E_img = Visualise_single(*abc, x, colour_function=propagate_BEM_pressure, colour_function_args={'scatterer':scatterer,'board':board,'path':path,"use_cache_H":cache_H,"p_ref":p_ref, "H":H } , res=res)
        E_phase = Visualise_single(*abc, x, colour_function=propagate_BEM_phase, colour_function_args={'scatterer':scatterer,'board':board,'path':path,"use_cache_H":cache_H,"p_ref":p_ref, "H":H } , res=res)

        GH_img = Visualise_single(*abc, x, colour_function=GH_prop, colour_function_args={'scatterer':scatterer,'board':board,'path':path,"use_cache_H":cache_H,"p_ref":p_ref, "H":H } , res=res)
        GH_phase = Visualise_single(*abc, x, colour_function=GH_prop_phase, colour_function_args={'scatterer':scatterer,'board':board,'path':path,"use_cache_H":cache_H,"p_ref":p_ref, "H":H } , res=res)

        F_img = Visualise_single(*abc, x, colour_function=propagate_abs, colour_function_args={'board':board,"p_ref":p_ref} , res=res)
        F_phase = Visualise_single(*abc, x, colour_function=propagate_phase, colour_function_args={'board':board,"p_ref":p_ref} , res=res)


        frame = [E_img, E_phase, GH_img, GH_phase, F_img, F_phase]
        frame = [img.real.to(torch.float32) for img in frame] + [i,]
        # frame = [img.real.to(torch.float32) for img in frame] + [d / wavelength]
    

        frames.append(frame)


vmax = 8000

for i,frame in enumerate(frames):
    print(i, end='\r')


    plt.figure()


    plt.subplot(2,3,1)
    plt.imshow(frame[0], vmax=8000, cmap='hot')
    plt.colorbar()

    plt.subplot(2,3,2)
    plt.imshow(frame[2], vmax=8000, cmap='hot')
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.imshow(frame[4], vmax=8000, cmap='hot')
    plt.colorbar()

    plt.subplot(2,3,4)
    plt.imshow(frame[1],cmap='hsv')
    plt.colorbar()

    plt.subplot(2,3,5)
    plt.imshow(frame[3],cmap='hsv')
    plt.colorbar()

    plt.subplot(2,3,6)
    plt.imshow(frame[5],cmap='hsv')
    plt.colorbar()

    plt.savefig(f"AcousTools_Examples/outputs/BEM_IB_GH_animate/Diameter{frame[-1]}_Parasite.png", dpi=1000)
    # plt.show()