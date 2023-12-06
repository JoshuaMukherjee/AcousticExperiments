from BEMLevitationObjectives import BEM_levitation_objective
from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas
from acoustools.Utilities import TRANSDUCERS

from acoustools.Solvers import wgs_wrapper

if __name__ == "__main__":

    path = "Media/Sphere-lam4.stl"
    scatterer = load_scatterer(path,dy=-0.06) #Make mesh at 0,0,0
    
    scale_to_diameter(scatterer,0.03)

    print(scatterer.bounds())
    # x1,x2,_,_,_,_ = scatterer.bounds()
    # vedo.show(scatterer)

    board = TRANSDUCERS

    centres = get_centres_as_points(scatterer)

    x = wgs_wrapper(centres)

    norms = get_normals_as_points(scatterer)

    areas = get_areas(scatterer)

    # centres = centres.expand((2,-1,-1))
    # norms  =norms.expand((2,-1,-1))
    areas = areas.expand((1,-1,-1))

    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas
    }


    BEM_levitation_objective(x, centres,board,**params)