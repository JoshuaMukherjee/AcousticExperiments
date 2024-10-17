from acoustools.Solvers import wgs, gspat
from acoustools.Utilities import TRANSDUCERS, create_points, propagate_abs, add_lev_sig
from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, get_edge_data
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure, BEM_gorkov_analytical
from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force


import torch, pickle

torch.manual_seed(1)

board = TRANSDUCERS

wall_paths = ["Media/flat-lam2.stl","Media/flat-lam2.stl"]
walls = load_multiple_scatterers(wall_paths,dxs=[-0.198/2,0.198/2],rotys=[90,-90]) #Make mesh at 0,0,0
walls.scale((1,19.3/12,22.5/12),reset=True,origin =False)
# print(walls)
walls.filename = scatterer_file_name(walls)
# print(walls)
get_edge_data(walls)

H = get_cache_or_compute_H(walls,board)


results = {
    'pressure':{
        'wgs': {},
        'gspat':{},
        'bem':{}
    },
    'gorkov':{
        'wgs': {},
        'gspat':{},
        'bem':{}
    },
    'force':{
        'wgs': {},
        'gspat':{},
        'bem':{}
    }
} 


N = [1,2,3,4,5,6,8,10,12,16,32,64]
M = 100

for m in range(M):
    
    print(m)
    for n in N:

        if m == 0:
            results['pressure']['wgs'][n] = []
            results['pressure']['gspat'][n] = []
            results['pressure']['bem'][n] = []

            results['gorkov']['wgs'][n] = []
            results['gorkov']['gspat'][n] = []
            results['gorkov']['bem'][n] = []

            results['force']['wgs'][n] = []
            results['force']['gspat'][n] = []
            results['force']['bem'][n] = []

        p = create_points(n,1)

    
        x_wgs = wgs(p, board=board, iter=200)

        x_gspat = gspat(p,board=board,iterations=200)

        E = compute_E(walls, p, board=board, H=H)
        x_bem = wgs(p, board=board, iter=200, A=E)


        def BEM_gorkov(p):
            return BEM_gorkov_analytical(x_bem, p, walls, board, H, E, path=None)

        def wgs_gorkov(p):
            U = gorkov_analytical(x_wgs,p, board)
            return U

        def gspat_gorkov(p):
            return gorkov_analytical(x_gspat,p,board)
            

        p_wgs =propagate_abs(x_wgs, p, board)
        p_gspat= propagate_abs(x_gspat, p, board)
        p_bem = propagate_BEM_pressure(x_bem,p,walls,board,E=E)


        x_wgs = add_lev_sig(x_wgs)
        x_gspat = add_lev_sig(x_gspat)
        x_bem = add_lev_sig(x_bem)

        U_wgs = gorkov_analytical(x_wgs,p, board)
        U_gspat = gorkov_analytical(x_gspat,p,board)
        U_bem = BEM_gorkov_analytical(x_bem, p, walls, board, E=E,H=H, path=None)
        
        F_wgs = torch.autograd.functional.jacobian(gorkov_analytical,(x_wgs,p, board))[1]
        F_gspat = torch.autograd.functional.jacobian(gorkov_analytical,(x_gspat,p, board))[1]
        F_bem = torch.autograd.functional.jacobian(BEM_gorkov,(p))[0]

        # stiff_wgs = torch.sum(torch.diagonal(torch.autograd.functional.hessian(wgs_gorkov,(p.real)).squeeze(),0))
        # stiff_gspat = torch.sum(torch.diagonal(torch.autograd.functional.hessian(gspat_gorkov,(p.real)).squeeze(),0))
        # stiff_bem = torch.sum(torch.diagonal(torch.autograd.functional.hessian(BEM_gorkov,(p.real)).squeeze(),0))

        results['pressure']['wgs'][n].append(p_wgs)
        results['pressure']['gspat'][n].append(p_gspat)
        results['pressure']['bem'][n].append(p_bem)

        results['gorkov']['wgs'][n].append(U_wgs)
        results['gorkov']['gspat'][n].append(U_gspat)
        results['gorkov']['bem'][n].append(U_bem)

        results['force']['wgs'][n].append(F_wgs)
        results['force']['gspat'][n].append(F_gspat)
        results['force']['bem'][n].append(F_bem)
        


pickle.dump(results, open('walls_quant.pth','wb'))
