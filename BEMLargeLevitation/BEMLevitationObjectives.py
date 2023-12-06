from acoustools.BEM import BEM_forward_model_grad
from acoustools.Gorkov import force_mesh

def BEM_levitation_objective(transducer_phases, points, board, targets=None, **objective_params):
    ''' Expects an element of objective_params named `norms` containing all normals in the same order as points 
    `areas` containing the areas for the cell containing each point
    `scatterer` containing the mesh to optimise around.'''


    scatterer = objective_params["scatterer"]
    norms = objective_params["norms"]
    areas = objective_params["areas"]

    params = {
        "scatterer":scatterer
    }

    force = force_mesh(transducer_phases,points,norms,areas,board,BEM_forward_model_grad,params)


    print(force)