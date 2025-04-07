if __name__ == "__main__":
    from acoustools.Mesh import load_scatterer, get_lines_from_plane,get_centre_of_mass_as_points, scale_to_diameter
    from acoustools.BEM import compute_E, propagate_BEM_pressure, compute_H
    from acoustools.Utilities import create_points, TOP_BOARD, propagate_abs, add_lev_sig
    from acoustools.Solvers import gspat, wgs
    from acoustools.Visualiser import Visualise
    import acoustools.Constants as c

    import torch, vedo, time


    '''Bk 2. Pg 140'''

    path = "../../BEMMedia"
    scatterer = load_scatterer(path+"/Sphere-lam2.stl",dy=-0.06,dz=-0.08)
    # scale_to_diameter(scatterer, 2*c.R)
    # scatterer = load_scatterer(path+"/Bunny-lam2.stl",dz=-0.10, rotz=90)
    # print(get_centre_of_mass_as_points(scatterer))
    # vedo.show(scatterer, axes =1)
    
    N=1
    B=1
    p = create_points(N,B,y=0,x=0,z=0)
    # p = create_points(N,B,y=0)
    
    # E = compute_E(scatterer, p, TOP_BOARD,path=path,use_cache_H=False)
    H = compute_H(scatterer, TOP_BOARD)
    E, F, G, H = compute_E(scatterer, p, TOP_BOARD,path=path,use_cache_H=False,return_components=True,H=H)

    start_gspat = time.time_ns()
    x = gspat(p,board=TOP_BOARD,A=E, iterations=200)
    end_gspat = time.time_ns()

    start_wgs = time.time_ns()
    x_wgs = wgs(p,board=TOP_BOARD,A=E, iter=200)
    end_wgs = time.time_ns()

    print("GS-PAT time:",(end_gspat-start_gspat)/1e9,'s')
    print("WGS time:",(end_wgs-start_wgs)/1e9,'s')


    # print(x.shape)

    x = add_lev_sig(x, TOP_BOARD, mode='Twin' )
    
    A = torch.tensor((-0.12,0, 0.12))
    B = torch.tensor((0.12,0, 0.12))
    C = torch.tensor((-0.12,0, -0.12))
    normal = (0,1,0)
    origin = (0,0,0)

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}

    # Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure,propagate_abs],colour_function_args=[{"scatterer":scatterer,"board":TOP_BOARD,"path":path,'H':H},{"board":TOP_BOARD}],vmax=8621, show=True,res=[256,256])
    Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":scatterer,"board":TOP_BOARD,"path":path,'H':H}],vmax=4000, show=True,res=[200,200])

