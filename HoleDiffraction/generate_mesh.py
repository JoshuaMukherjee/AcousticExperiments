import trimesh, vedo, random

def fractal_mesh(cube: trimesh.Trimesh, size:int) -> trimesh.Trimesh:
    
    facets =cube.facets
    print(facets)
    boundaries = cube.facets_boundary
    origins = cube.facets_origin

    idx = random.randint(0,len(facets)-1)

    origin = origins[idx]

    normal = cube.face_normals[idx]
    print(normal)

    transform = [[1,0,0,normal[0] * (size/2 + origin[0])],
                 [0,1,0,normal[1] * (size/2 + origin[1])],
                 [0,0,1,normal[2] * (size/2 + origin[2])],
                 [0,0,0,1]]
    new_cube = trimesh.creation.box((size + size*normal[0],size+ size*normal[1],size+ size*normal[2]), transform=transform)
    cube = trimesh.util.concatenate( [ cube, new_cube ] )


    # cube.__add__(new_cube)
    
    return cube



if __name__ == '__main__':
    N = 4
    size =1
    cube = trimesh.creation.box((size,size,size))
    for i in range(N):
        cube = fractal_mesh(cube, size/2)
        size = size/2
    vedo.show(cube)
