import pickle, torch

from acoustools.Visualiser import Visualise_mesh
from acoustools.Force import force_mesh
from acoustools.Mesh import load_multiple_scatterers,scatterer_file_name, get_edge_data, load_scatterer, scale_to_diameter, \
                            merge_scatterers, get_centres_as_points, get_normals_as_points, get_areas
from acoustools.BEM import get_cache_or_compute_H_gradients, get_cache_or_compute_H,grad_H
from acoustools.Utilities import DTYPE, TRANSDUCERS
import acoustools.Constants as c
import matplotlib.pyplot as plt

board = TRANSDUCERS


x = pickle.load(open('BEMLargeLevitation/Paths/holo.pth','rb'))

wall_paths = ["Media/flat-lam2.stl","Media/flat-lam2.stl"]
walls = load_multiple_scatterers(wall_paths,dxs=[-0.198/2,0.198/2],rotys=[90,-90]) #Make mesh at 0,0,0
walls.scale((1,19.3/12,22.5/12),reset=True,origin =False)
# print(walls)
walls.filename = scatterer_file_name(walls)
# print(walls)
get_edge_data(walls)



ball_path = "Media/Sphere-lam2.stl"
ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
scale_to_diameter(ball,0.02)

get_edge_data(ball)
# scale_to_diameter(ball, Constants.R*2)

scatterer = merge_scatterers(ball, walls)


Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=True)
Hx_ball, Hy_ball, Hz_ball = get_cache_or_compute_H_gradients(ball, board,print_lines=True)



px = (Hx@x).squeeze_(2).unsqueeze_(0)
py = (Hy@x).squeeze_(2).unsqueeze_(0)
pz = (Hz@x).squeeze_(2).unsqueeze_(0)

px[px.isnan()] = 0
py[py.isnan()] = 0
pz[pz.isnan()] = 0

k1 = 1/ (2*c.p_0*(c.c_0**2))
k2 = 1/ (c.k**2)


grad = torch.cat((px,py,pz),dim=1).to(DTYPE)
grad_norm = torch.norm(grad,2,dim=1)**2
g = k2 * grad_norm[:,:1728].unsqueeze_(2)

# exit()
H_ball = get_cache_or_compute_H(ball, board)
p = torch.abs(H_ball@x)**2

norms = get_normals_as_points(ball)
areas = get_areas(ball)

a = areas.unsqueeze(2)

force = (-1/2 * k1 * (p + g) * a)
force = force.permute(0,2,1).expand(1,3,-1) * norms
print(force.shape, norms.shape)
print(torch.sum(force,dim=2))

print(g, p)


fig = plt.figure()
Visualise_mesh(ball,g, fig=fig, subplot=131, show=False)
Visualise_mesh(ball,p, fig=fig, subplot=132, show=False)
Visualise_mesh(ball,torch.real(force), fig=fig, subplot=133)

plt.show()