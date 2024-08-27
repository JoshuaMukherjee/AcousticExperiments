
import trimesh
import random
import numpy as np
import itertools

def extrude(icoso:trimesh.Trimesh, top_faces:trimesh.caching.TrackedArray,side_faces:list, size=1):
    face_id = random.randint(0,len(top_faces)-1)
    face = np.array([v for v in top_faces[face_id]])
    normal = icoso.face_normals[face_id]

    N_vert = len(icoso.vertices)
   
    mask = np.ones_like(icoso.faces)
    mask[face_id,:] = 0
    mask = mask.any(axis=1)
    mask = np.bool_(mask)
    
    # icoso.update_faces(mask)
    # top_faces = top_faces[mask]

    
    vertices = [icoso.vertices[v] for v in face]
    
    new_vertices = [v+normal*size for v in vertices]
    top_face = [N_vert, N_vert+1, N_vert+2]
    top_faces[face_id] = top_face

    icoso.vertices = np.vstack([icoso.vertices, new_vertices])

    for (a,b) in itertools.combinations(top_face,2):
        a_ind = np.where(np.array(top_face) == a)
        b_ind = np.where(np.array(top_face)  == b)

        face_1 = [a,face[a_ind].item(),b]
        face_2 = [face[a_ind].item(),face[b_ind].item(),b]
        side_faces.append(face_1)
        side_faces.append(face_2)


    icoso.faces = np.vstack([top_faces, side_faces])
    trimesh.repair.fix_normals(icoso, True)
    trimesh.repair.fix_inversion(icoso, True)
    trimesh.repair.fill_holes(icoso)



if __name__ == '__main__':
    
    for set in range(10):
        icoso = trimesh.creation.icosahedron()
        scale = 0.02
        transform = [[scale,0,0,0],
                    [0,scale,0,0],
                    [0,0,scale,0],
                    [0,0,0,1]]
        icoso.apply_transform(transform)
        top_faces= icoso.faces
        side_faces = []
        for n in range(30):
            extrude(icoso,top_faces,side_faces, size=0.01)
            # print('Exporting')
            print(set, n)
            m = icoso.subdivide_to_size(0.00865)
            stl = trimesh.exchange.stl.export_stl_ascii(m)
            f = open(f'MeshBranches/Meshes/icoso/set{set}/m{n}.stl','w')
            f.write(stl)
            f.close()
