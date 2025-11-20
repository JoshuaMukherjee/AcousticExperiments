from acoustools.Mesh import load_scatterer, rotate, centre_scatterer, scale_to_diameter

import vedo

path = '../BEMMedia/'
block_pth = path + 'block-lam4.stl'

block = load_scatterer(block_pth)
scale_to_diameter(block, 0.02)
centre_scatterer(block)

block_rot_x = block.clone()
rotate(block_rot_x, (1,0,0), 45, rotate_around_COM=True)

block_rot_y = block.clone()
rotate(block_rot_y, (0,1,0), 45, rotate_around_COM=True)

block_rot_z = block.clone()
rotate(block_rot_z, (0,0,1), 45, rotate_around_COM=True)




vedo.show((block, block_rot_z),new=True )