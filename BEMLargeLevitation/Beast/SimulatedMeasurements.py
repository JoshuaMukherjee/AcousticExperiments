if __name__ == '__main__':
    from acoustools.Utilities import read_phases_from_file, TRANSDUCERS
    from acoustools.Visualiser import Visualise_single
    from acoustools.BEM import propagate_BEM_pressure, get_cache_or_compute_H
    from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name, load_scatterer, scale_to_diameter, merge_scatterers, get_areas
    from acoustools.read_beast_file import read_beast_file

    import matplotlib.pyplot as plt

    import torch
    import numpy as np

    board = TRANSDUCERS

    wall_paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.175/2,0.175/2],rotys=[90,-90]) #Make mesh at 0,0,0
    walls.scale((1,19/12,19/12),reset=True,origin =False)
    walls.filename = scatterer_file_name(walls)
    


    ball_path = "Media/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.02)
    # scale_to_diameter(ball, Constants.R*2)

    scatterer = merge_scatterers(ball, walls)
    print(scatterer)

    H = get_cache_or_compute_H(walls,board,print_lines=True)

    print('reading...')
    path = r"C:\Users\joshu\Documents\AcousticExperiments\BEMLargeLevitation\Paths\spherelevBeast28-2-24-Test1.csv"
    x = read_phases_from_file(path)

    start = -0.02
    d = 0.04
    
    xz_dy = 0
    xy_dz = -0.00175
    yz_dx = 0

    As = [torch.tensor((start,xz_dy, start+d)),     torch.tensor((start,start+d,xy_dz)),          torch.tensor((yz_dx,start, start+d))]
    Bs = [torch.tensor((start+d,xz_dy, start+d)),   torch.tensor((start+d,start+d, xy_dz)),     torch.tensor((yz_dx,start+d, start+d))]
    Cs = [torch.tensor((start,xz_dy, start)),       torch.tensor((start,start, xy_dz)),         torch.tensor((yz_dx,start, start))]
    
    fig = plt.figure()
    axes = []
    titles = ['xz','xy','yz']
    for i in range(3):
        A = As[i]
        B = Bs[i]
        C = Cs[i]

        ax = fig.add_subplot(2,3,i+1)
        axes.append(ax)
        mat = Visualise_single(A,B,C,x,colour_function=propagate_BEM_pressure,colour_function_args={"H":H,"scatterer":walls,"board":board},res=(40,40)).cpu().detach().numpy()
        im = ax.matshow(mat,cmap='hot',vmax=5000)
        if i ==0: ax.set_ylabel('Simulated')
        fig.colorbar(im)
        ax.set_title(titles[i])
    
    files = [r"C:\Users\joshu\Documents\Figs\BEMLargeLevitation\Beast28-2-24\Scan DATE_28 2 2024; TIME_11 36.csv",  r"C:\Users\joshu\Documents\Figs\BEMLargeLevitation\Beast28-2-24\Scan DATE_28 2 2024; TIME_12 6.csv", r"C:\Users\joshu\Documents\Figs\BEMLargeLevitation\Beast28-2-24\Scan DATE_28 2 2024; TIME_12 53.csv"]
    min_xs = [55,55,75]
    min_ys = [175,155,155]
    min_zs = [57,77,57]

    step_xs = [40,40,0]
    step_ys = [0,40,40]
    step_zs = [40,0,40]

    mode = "Peak"             # Mode can be "RMS" or "Peak"
    amp_setting=1.0/1000      # amp_setting is how many volts per pascal, native 0dB gain on amp is 1mv/Pa for lowest noise.
    mic_gain_correction=0.81  # mic_gain_correction is gain due to incident angle on microphone, default is for 90 degrees. See below for reference values extracted from datasheet graph



    for i in range(3):
        amp, phase = read_beast_file(files[i], mode, amp_setting, mic_gain_correction)
        min_x = min_xs[i]
        min_y = min_ys[i]
        min_z = min_zs[i]

        step_x = step_xs[i]
        step_y = step_ys[i]
        step_z = step_zs[i]

        amp_filtered = amp[min_x:min_x+step_x+1, min_y:min_y+step_y+1, min_z:min_z+step_z+1]

        amp_squeeze = amp_filtered.squeeze()
        amp_squeeze = amp_squeeze.T
        if i == 1:
            amp_squeeze = np.flipud(amp_squeeze)
        elif i == 2:
            amp_squeeze = np.fliplr(amp_squeeze)

        ax = fig.add_subplot(2,3,3+i+1)
        im = ax.matshow(amp_squeeze,cmap='hot',vmax=5000)
        if i ==0: ax.set_ylabel('Measured ('+mode+')')
        fig.colorbar(im)



    # fig.subplots_adjust(right=0.8)
    # # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, ax=axes)

    plt.show()
