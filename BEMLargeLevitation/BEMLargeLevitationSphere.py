from BEMLevitationObjectives import BEM_levitation_objective
from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas, get_lines_from_plane
from acoustools.Utilities import TRANSDUCERS, propagate_abs, get_convert_indexes
from acoustools.BEM import compute_H, grad_H
from acoustools.Visualiser import Visualise
from acoustools.Solvers import gradient_descent_solver, wgs_wrapper, wgs_batch
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Gorkov import force_mesh

import torch

if __name__ == "__main__":

    path = "Media/Sphere-lam1.stl"
    scatterer = load_scatterer(path,dy=-0.06) #Make mesh at 0,0,0
    
    scale_to_diameter(scatterer,0.04)

    print(scatterer.bounds())
    # x1,x2,_,_,_,_ = scatterer.bounds()
    # vedo.show(scatterer)

    board = TRANSDUCERS

    centres = get_centres_as_points(scatterer)

    H = compute_H(scatterer, board)
    # Hx, Hy, Hz = grad_H(scatterer,board)

    # x = wgs_wrapper(centres, A=H)

    # print(propagate_abs(x,centres,A=H))

    # x = wgs_wrapper(centres)

    norms = get_normals_as_points(scatterer)
    areas = get_areas(scatterer)

    # centres = centres.expand((2,-1,-1))
    # norms  =norms.expand((2,-1,-1))
    areas = areas.expand((1,-1,-1))

    



    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":0.0027*9.81
    }


    x = gradient_descent_solver(centres, BEM_levitation_objective,constrains=constrain_phase_only,objective_params=params,log=True,iters=400,lr=0.005)
    f = force_mesh(x,centres,norms,areas,board,grad_H,grad_function_args={"scatterer":scatterer })
    print(torch.sum(f,dim=2), 0.0027*9.81)

    A = torch.tensor((-0.07,0, 0.07))
    B = torch.tensor((0.07,0, 0.07))
    C = torch.tensor((-0.07,0, -0.07))

    origin = (0,0,-0.07)
    normal = (0,1,0)

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}

   

    # Visualise(A,B,C,x,colour_functions=[propagate_abs], add_lines_functions=[get_lines_from_plane],add_line_args=[line_params])
    
    # exit()


    FLIP_INDEXES = get_convert_indexes()
    row = torch.angle(x).squeeze_()
    row_flip = row[FLIP_INDEXES]


    num_frames = 1
    num_transducers = 512

    output_f = open("./BEMLargeLevitation/Paths/spherelev"+".csv","w")
    output_f.write(str(num_frames)+","+str(num_transducers)+"\n")
    for i,phase in enumerate(row_flip):
                output_f.write(str(phase.item()))
                if i < 511:
                    output_f.write(",")
                else:
                    output_f.write("\n")

    output_f.close()