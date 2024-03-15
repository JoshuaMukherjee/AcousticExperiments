if __name__ == '__main__':
    from acoustools.Utilities import read_phases_from_file, TRANSDUCERS, get_rows_in
    from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, translate, rotate, load_scatterer, scale_to_diameter, merge_scatterers, get_centres_as_points, get_normals_as_points, get_areas
    from acoustools.Force import force_mesh
    from acoustools.BEM import get_cache_or_compute_H_gradients, get_cache_or_compute_H

    import matplotlib.pyplot as plt

    import torch, random, vedo

    board = TRANSDUCERS

    wall_paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.175/2,0.175/2],rotys=[90,-90]) #Make mesh at 0,0,0
    walls.scale((1,19/12,19/12),reset=True,origin =False)
    walls.filename = scatterer_file_name(walls)

    x = read_phases_from_file(r'C:\Users\joshu\Documents\AcousticExperiments\BEMLargeLevitation\Paths\spherelevSideways.csv')

    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)
    

    max_angle = torch.pi/16 #approx 10 degrees
    max_translation =0.005

    time_stamps = 10000

    ball_path = "Media/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.02)

    scatterer = merge_scatterers(ball, walls)
    scatterer_cells = get_centres_as_points(scatterer)
    ball_cells = get_centres_as_points(ball)
    mask = get_rows_in(scatterer_cells,ball_cells, expand=False).squeeze_()


    fxs = []
    fys = []
    fzs = []

    for i in range(time_stamps):
        print(i, end=' ',flush=True)
        walls_moved = walls.clone()
        dx = random.randrange(-1,1) * max_translation
        dy = random.randrange(-1,1) * max_translation
        dz = random.randrange(-1,1) * max_translation
        translate(walls_moved,dx,dy,dz)
        
        rot_x = random.randrange(-1,1) * max_angle
        rot_y = random.randrange(-1,1) * max_angle
        rot_z = random.randrange(-1,1) * max_angle
        rotate(walls_moved,[1,0,0],rot_x)
        rotate(walls_moved,[0,1,0],rot_y)
        rotate(walls_moved,[0,0,1],rot_z)

        scatterer = merge_scatterers(ball, walls_moved)
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

        fxs.append(torch.sum(force_x).cpu().detach())
        fys.append(torch.sum(force_y).cpu().detach())
        fzs.append(torch.sum(force_z).cpu().detach())


    plt.hist(fxs,histtype=u'step',label='$F_x$',bins=100)
    plt.hist(fys,histtype=u'step',label='$F_y$',bins=100)
    plt.hist(fzs,histtype=u'step',label='$F_z$',bins=100)
    plt.legend()
    plt.show()
