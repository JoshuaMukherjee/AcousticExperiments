if __name__ == '__main__':
    from acoustools.Utilities import read_phases_from_file, TRANSDUCERS, get_rows_in
    from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, translate, rotate, load_scatterer, scale_to_diameter, merge_scatterers, get_centres_as_points, get_normals_as_points, get_areas, get_centre_of_mass_as_points
    from acoustools.Force import force_mesh
    from acoustools.BEM import get_cache_or_compute_H_gradients, get_cache_or_compute_H

    import matplotlib.pyplot as plt

    import torch, random, vedo
    import numpy as np

    board = TRANSDUCERS

    left_wall = load_scatterer("Media/flat-lam1.stl",dx=-0.175/2, roty=90)
    left_wall.scale((1,19/12,19/12),reset=True,origin =False)
    left_wall.filename = scatterer_file_name(left_wall)

    right_wall = load_scatterer("Media/flat-lam1.stl",dx=0.175/2, roty=-90)
    right_wall.scale((1,19/12,19/12),reset=True,origin =False)
    right_wall.filename = scatterer_file_name(right_wall)    

    x = read_phases_from_file(r'C:\Users\joshu\Documents\AcousticExperiments\BEMLargeLevitation\Paths\spherelevSideways.csv')

    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    ball_path = "Media/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.02)

    scatterer = merge_scatterers(ball, left_wall, right_wall)
    scatterer_cells = get_centres_as_points(scatterer)
    ball_cells = get_centres_as_points(ball)
    mask = get_rows_in(scatterer_cells,ball_cells, expand=False).squeeze_()

    steps = 200
    max_angle = 20
    max_translation =0.02

    RotXFs = [[],[],[]]
    RotYFs = [[],[],[]]
    RotZFs = [[],[],[]]
    dXFs = [[],[],[]]
    dYFs = [[],[],[]]
    dZFs = [[],[],[]]

    def get_force(left_wall,right_wall, ball, trans=(0,0,0), rot=(0,0,0)):
        walls_moved_l = left_wall.clone()
        walls_moved_r = right_wall.clone()


        dx,dy,dz=  trans
        translate(walls_moved_l,dx,dy,dz)
        rot_x, rot_y, rot_z = rot   
        rotate(walls_moved_l,[1,0,0],rot_x,rotate_around_COM=True)
        rotate(walls_moved_l,[0,1,0],rot_y,rotate_around_COM=True)
        rotate(walls_moved_l,[0,0,1],rot_z,rotate_around_COM=True)

        dx,dy,dz=  trans
        translate(walls_moved_r,dx,dy,dz)
        rot_x, rot_y, rot_z = rot   
        rotate(walls_moved_r,[1,0,0],rot_x,rotate_around_COM=True)
        rotate(walls_moved_r,[0,1,0],rot_y,rotate_around_COM=True)
        rotate(walls_moved_r,[0,0,1],rot_z,rotate_around_COM=True)

        scatterer = merge_scatterers(ball, walls_moved_r, walls_moved_l)



        centres = get_centres_as_points(scatterer)
        norms = get_normals_as_points(scatterer)
        areas = get_areas(scatterer)
        areas = areas.expand((1,-1,-1))

        Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=False)
        H = get_cache_or_compute_H(scatterer,board,print_lines=False)

        force = force_mesh(x,centres,norms,areas,board,None,None,Ax=Hx, Ay=Hy, Az=Hz,F=H)

        force_x = force[:,0,:][:,mask]
        force_y = force[:,1,:][:,mask]
        force_z = force[:,2,:][:,mask]

        fx = torch.sum(force_x).cpu().detach()
        fy = torch.sum(force_y).cpu().detach()
        fz = torch.sum(force_z).cpu().detach()

        return fx, fy, fz
    
    MIN_F = 10000
    MAX_F = -10000
    
    angles = []
    for i in range(steps):
        print(i, end=' ',flush=True)
        start = -1*max_angle
        end = max_angle

        angle = start + ((end-start)/steps)*i
        angles.append(angle)

        fx, fy, fz = get_force(left_wall,right_wall, ball, (0,0,0),(angle,0,0))

        RotXFs[0].append(fx)
        RotXFs[1].append(fy)
        RotXFs[2].append(fz)

    MIN_F = min([x for xs in RotXFs for x in xs]) if min([x for xs in RotXFs for x in xs]) < MIN_F else MIN_F
    MAX_F = max([x for xs in RotXFs for x in xs]) if max([x for xs in RotXFs for x in xs]) > MAX_F else MAX_F
    
    plt.subplot(2,3,1)
    plt.plot(angles, RotXFs[0],label='$F_x$')
    plt.plot(angles, RotXFs[1],label='$F_y$')
    plt.plot(angles, RotXFs[2],label='$F_z$')
    plt.title('Rotation around x-axis')
    plt.legend()
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Force (N)')

    angles = []
    for i in range(steps):
        print(i, end=' ',flush=True)
        start = -1*max_angle
        end = max_angle

        angle = start + ((end-start)/steps)*i
        angles.append(angle)

        fx, fy, fz = get_force(left_wall,right_wall, ball, (0,0,0),(0,angle,0))

        RotYFs[0].append(fx)
        RotYFs[1].append(fy)
        RotYFs[2].append(fz)

    MIN_F = min([x for xs in RotYFs for x in xs]) if min([x for xs in RotYFs for x in xs]) < MIN_F else MIN_F
    MAX_F = max([x for xs in RotYFs for x in xs]) if max([x for xs in RotYFs for x in xs]) > MAX_F else MAX_F

    plt.subplot(2,3,2)
    plt.plot(angles, RotYFs[0],label='$F_x$')
    plt.plot(angles, RotYFs[1],label='$F_y$')
    plt.plot(angles, RotYFs[2],label='$F_z$')
    plt.title('Rotation around y-axis')
    plt.legend()
    plt.xlabel('Angle (degrees)')
    # plt.ylabel('Force (N)')



    angles = []
    for i in range(steps):
        print(i, end=' ',flush=True)
        start = -1*max_angle
        end = max_angle

        angle = start + ((end-start)/steps)*i
        angles.append(angle)

        fx, fy, fz = get_force(left_wall,right_wall, ball, (0,0,0),(0,0,angle))

        RotZFs[0].append(fx)
        RotZFs[1].append(fy)
        RotZFs[2].append(fz)

    MIN_F = min([x for xs in RotZFs for x in xs]) if min([x for xs in RotZFs for x in xs]) < MIN_F else MIN_F
    MAX_F = max([x for xs in RotZFs for x in xs]) if max([x for xs in RotZFs for x in xs]) > MAX_F else MAX_F

    plt.subplot(2,3,3)
    plt.plot(angles, RotZFs[0],label='$F_x$')
    plt.plot(angles, RotZFs[1],label='$F_y$')
    plt.plot(angles, RotZFs[2],label='$F_z$')
    plt.title('Rotation around z-axis')
    plt.legend()
    plt.xlabel('Angle (degrees)')
    # plt.ylabel('Force (N)')
    
    transes = []
    for i in range(steps):
        print(i, end=' ',flush=True)
        start = -1*max_translation
        end = max_translation

        trans = start + ((end-start)/steps)*i
        transes.append(trans)

        fx, fy, fz = get_force(left_wall,right_wall, ball, (trans,0,0),(0,0,0))

        dXFs[0].append(fx)
        dXFs[1].append(fy)
        dXFs[2].append(fz)

    MIN_F = min([x for xs in dXFs for x in xs]) if min([x for xs in dXFs for x in xs]) < MIN_F else MIN_F
    MAX_F = max([x for xs in dXFs for x in xs]) if max([x for xs in dXFs for x in xs]) > MAX_F else MAX_F

    plt.subplot(2,3,4)
    plt.plot(transes, dXFs[0],label='$F_x$')
    plt.plot(transes, dXFs[1],label='$F_y$')
    plt.plot(transes, dXFs[2],label='$F_z$')
    plt.title('Translation in x-direction')
    plt.legend()
    plt.xlabel('dx (m)')
    plt.ylabel('Force (N)')

    transes = []
    for i in range(steps):
        print(i, end=' ',flush=True)
        start = -1*max_translation
        end = max_translation

        trans = start + ((end-start)/steps)*i
        transes.append(trans)

        fx, fy, fz = get_force(left_wall,right_wall, ball, (0,trans,0),(0,0,0))

        dYFs[0].append(fx)
        dYFs[1].append(fy)
        dYFs[2].append(fz)

    MIN_F = min([x for xs in dYFs for x in xs]) if min([x for xs in dYFs for x in xs]) < MIN_F else MIN_F
    MAX_F = max([x for xs in dYFs for x in xs]) if max([x for xs in dYFs for x in xs]) > MAX_F else MAX_F

    plt.subplot(2,3,5)
    plt.plot(transes, dYFs[0],label='$F_x$')
    plt.plot(transes, dYFs[1],label='$F_y$')
    plt.plot(transes, dYFs[2],label='$F_z$')
    plt.title('Translation in y-direction')
    plt.legend()
    plt.xlabel('dy (m)')

    transes = []
    for i in range(steps):
        print(i, end=' ',flush=True)
        start = -1*max_translation
        end = max_translation

        trans = start + ((end-start)/steps)*i
        transes.append(trans)

        fx, fy, fz = get_force(left_wall,right_wall, ball, (0,0, trans),(0,0,0))

        dZFs[0].append(fx)
        dZFs[1].append(fy)
        dZFs[2].append(fz)

    MIN_F = min([x for xs in dZFs for x in xs]) if min([x for xs in dZFs for x in xs]) < MIN_F else MIN_F
    MAX_F = max([x for xs in dZFs for x in xs]) if max([x for xs in dZFs for x in xs]) > MAX_F else MAX_F

    plt.subplot(2,3,6)
    plt.plot(transes, dZFs[0],label='$F_x$')
    plt.plot(transes, dZFs[1],label='$F_y$')
    plt.plot(transes, dZFs[2],label='$F_z$')
    plt.title('Translation in z-direction')
    plt.legend()
    plt.xlabel('dz (m)')
    
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.ylim(MIN_F*1.1,MAX_F*1.1)

    plt.show()

    