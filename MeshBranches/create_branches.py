import trimesh, random
import numpy as np
import itertools
import vedo

from numpy.linalg import norm



from acoustools.Mesh import get_edge_data

def branch_mesh(mesh:trimesh.Trimesh, size:float=1):
    idx = random.randint(0,len(mesh.faces)-1)
    N_faces = len(mesh.faces)
    N_vert = len(mesh.vertices)

    
    face = mesh.faces[idx]
    normal = mesh.face_normals[idx]
    vertices = [mesh.vertices[v] for v in face]
    
    new_vertices = [v+normal*size for v in vertices]
    top_face = [N_vert, N_vert+1, N_vert+2]

    mesh.vertices = np.vstack([mesh.vertices, new_vertices])

    new_faces = [top_face]
    for (a,b) in itertools.combinations(top_face,2):
        a_ind = np.where(np.array(top_face) == a)
        b_ind = np.where(np.array(top_face)  == b)



        face_1 = [a,face[a_ind].item(),b]
        face_2 = [face[a_ind].item(),face[b_ind].item(),b]
        new_faces.append(face_1)
        new_faces.append(face_2)


    mesh.faces = np.vstack([mesh.faces, new_faces])
    trimesh.repair.fix_normals(mesh, True)
    trimesh.repair.fix_inversion(mesh, True)
    trimesh.repair.fill_holes(mesh)

    
    # mesh = mesh.convex_hulljj



if __name__ == '__main__':

    mesh = trimesh.creation.icosahedron()
    scale = 0.01
    transform = [[scale,0,0,0],
                 [0,scale,0,0],
                 [0,0,scale,0],
                 [0,0,0,1]]
    mesh.apply_transform(transform)

    
    # mesh = vedo.Mesh(mesh)


    # mesh = trimesh.creation.box((1,1,1))
    N = 50
    lam = 1
    for j in range(5):
        mesh = trimesh.creation.icosahedron()

        for i in range(N):
            branch_mesh(mesh)
            print(i)   


            print('Subdividing')
            m = m.subdivide_to_size(0.00865/lam)
            trimesh.smoothing.filter_humphrey(m)
            print('Exporting')
            print(m)
            stl = trimesh.exchange.stl.export_stl_ascii(m)
            f = open(f'MeshBranches/Meshes/set{j}/m{i}.stl','w')
            f.write(stl)
            f.close()
            exit()
        

