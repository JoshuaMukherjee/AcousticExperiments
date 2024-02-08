import torch
from acoustools.Mesh import translate, merge_scatterers, get_centre_of_mass_as_points, get_centres_as_points
from acoustools.BEM import get_cache_or_compute_H, get_cache_or_compute_H_gradients, compute_E, BEM_forward_model_grad

def get_indexes_subsample(N, centres):
    indexes = torch.zeros(centres.shape[2]).to(bool)
    mask = torch.randperm(centres.shape[2])[:N]
    indexes[mask] = True

    return indexes


def get_H_for_fin_diffs(start,end, scatterers, board, steps=1, path="Media",print_lines=False, use_cache=True):
    direction = (end - start) / steps  

    translate(scatterers[0], start[0].item() - direction[0].item(), start[1].item() - direction[1].item(), start[2].item() - direction[2].item())
    scatterer = merge_scatterers(*scatterers)
    
    Hs = []
    Hxs = []
    Hys = []
    Hzs = []
    
    for i in range(steps+1):
        if print_lines:
            print(i)
        
        
        translate(scatterers[0], direction[0].item(), direction[1].item(), direction[2].item())
        scatterer = merge_scatterers(*scatterers)

        H = get_cache_or_compute_H(scatterer, board, path=path, print_lines=print_lines, use_cache_H=use_cache)
        Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board, path=path, print_lines=print_lines, use_cache_H_grad=use_cache)
    
        Hs.append(H)
        Hxs.append(Hx)
        Hys.append(Hy)
        Hzs.append(Hz)
    
    return Hs, Hxs, Hys, Hzs

def  get_E_for_fin_diffs(start,end, scatterers, board, steps=1, path="Media",print_lines=False, use_cache=True, normal_scale=0.001):
    direction = (end - start) / steps  

    translate(scatterers[0], start[0].item() - direction[0].item(), start[1].item() - direction[1].item(), start[2].item() - direction[2].item())
    scatterer = merge_scatterers(*scatterers)
    
    Es = []
    Exs = []
    Eys = []
    Ezs = []
    
    for i in range(steps+1):
        if print_lines:
            print(i)
        
        
        translate(scatterers[0], direction[0].item(), direction[1].item(), direction[2].item())
        scatterer = merge_scatterers(*scatterers)
        centres = get_centres_as_points(scatterer, add_normals=True, normal_scale=normal_scale)

        E = compute_E(scatterer,centres,board,use_cache,print_lines,path=path)
        Ex, Ey, Ez = BEM_forward_model_grad(centres, scatterer,board, use_cache, print_lines,path=path)
    
        Es.append(E)
        Exs.append(Ex)
        Eys.append(Ey)
        Ezs.append(Ez)
    
    return Es, Exs, Eys, Ezs
