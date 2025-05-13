if __name__ == "__main__":
    from acoustools.Force import force_mesh, compute_force, force_fin_diff
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, TRANSDUCERS
    from acoustools.Solvers import wgs
    from acoustools.Mesh import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_areas, get_weight, scale_to_diameter, get_centre_of_mass_as_points,get_lines_from_plane
    from acoustools.BEM import compute_E, BEM_forward_model_grad, propagate_BEM_pressure, BEM_gorkov_analytical
    import acoustools.Constants as c 
    from acoustools.Visualiser import Visualise, ABC, Visualise_single

    import vedo, torch

    import matplotlib.pyplot as plt

    board = TRANSDUCERS

    p = create_points(1,1,0,0,0)

    x = wgs(p, board=board)
    x = add_lev_sig(x, mode='Twin')

    force = force_fin_diff(x, p, board=board,stepsize=c.wavelength/8)

    print(force.shape)

    print(force)

    res = (100,100)
    xz = Visualise_single(*ABC(0.03), x, res=res)
    yz = Visualise_single(*ABC(0.03, plane='yz'), x, res=res)

    plt.subplot(1,3,1)
    plt.imshow(xz,cmap='hot')

    plt.subplot(1,3,2)
    plt.imshow(yz,cmap='hot')

    plt.subplot(1,3,3)
    plt.imshow(xz-yz,cmap='hot')


    plt.show()