from acoustools.Mesh import load_scatterer, scale_to_diameter, centre_scatterer, get_centre_of_mass_as_points, translate
from acoustools.Constants import wavelength
from acoustools.Gorkov import get_finite_diff_points_all_axis
from acoustools.Force import compute_force
from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Solvers import wgs
from acoustools.BEM import force_mesh_surface

path = "../BEMMedia"

board = TRANSDUCERS
pt = create_points(1,1)
x = wgs(pt, board=board)

start_pos = -2 * wavelength
end_pos = 2 * wavelength
N = 16

d = wavelength/2


ps = []
Fs = []
F_surfaces= []

step = wavelength/10

sphere_pth =  path+"/Sphere-solidworks-lam2.stl"

axis = 'X'


def force_gradient(point, axis='Z'):
    ps = get_finite_diff_points_all_axis(point, axis=axis, stepsize=step)
    _,_,F = compute_force(x, ps, board=board, return_components=True)
    return (F[:,1] - F[:,2]) / 2*step

def force_surface_gradient(point, axis='Z'):
    ps = get_finite_diff_points_all_axis(point, axis=axis, stepsize=step)
    fs = []
    for i in range(1,ps.shape[2]):
        sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
        scale_to_diameter(sphere,d)
        centre_scatterer(sphere)
        translate(sphere, dx=ps[:,0,i].item(), dy=ps[:,1,i].item(), dz=ps[:,2,i].item())
        _,_,F_surface = force_mesh_surface(x, sphere, board,return_components=True, path=path)
        fs.append(F_surface)
    
    return (fs[0] - fs[1]) / 2*step


for i in range(N):
    print(i, end='\r')

    p = ((end_pos - start_pos) / N)* i
    ps.append(p)

    sphere = load_scatterer(sphere_pth) #Make mesh at 0,0,0
    scale_to_diameter(sphere,d)
    centre_scatterer(sphere)
    translate(sphere, dx=start_pos + p)
    com = get_centre_of_mass_as_points(sphere)


    F = force_gradient(com)
    F_surface = force_surface_gradient(com)

    # fd_points = get_finite_diff_points_all_axis(com, stepsize=wavelength/16, axis='Z')

    

    Fs.append(F.item())
    F_surfaces.append(F_surface.item())


import matplotlib.pyplot as plt

plt.plot(ps, Fs,label=r'${\nabla_z F_U}$')
plt.plot(ps, F_surfaces,label=r'${\nabla_z F_{ARF}}$')

plt.ylabel(r'$\nabla_z F (Nm^{-1})$')
plt.xlabel('Displacement (m)')
plt.legend()
plt.show()