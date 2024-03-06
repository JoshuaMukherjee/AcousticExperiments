if __name__ == '__main__':
        from acoustools.read_beast_file import read_beast_file

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        import torch

        from acoustools.Visualiser import Visualise_single
        from acoustools.BEM import propagate_BEM_pressure, get_cache_or_compute_H
        from acoustools.Utilities import read_phases_from_file, TRANSDUCERS
        from acoustools.Mesh import load_multiple_scatterers, scatterer_file_name

        board = TRANSDUCERS

        min_x = 63
        min_y = 163
        min_z = 65

        step_x = 24
        step_y = 24
        step_z = 12

        start = -0.02
        d = 0.04

        xz_dy = -0.02
        steps= 0.04 / step_y


        print('reading...')
        path = r"C:\Users\joshu\Documents\AcousticExperiments\BEMLargeLevitation\Paths\spherelevBeast28-2-24-Test1.csv"
        x = read_phases_from_file(path)


        A = torch.tensor((start,xz_dy, start+d))
        B = torch.tensor((start+d,xz_dy, start+d))
        C = torch.tensor((start,xz_dy, start))

        wall_paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
        walls = load_multiple_scatterers(wall_paths,dxs=[-0.175/2,0.175/2],rotys=[90,-90]) #Make mesh at 0,0,0
        walls.scale((1,19/12,19/12),reset=True,origin =False)
        walls.filename = scatterer_file_name(walls)

        H = get_cache_or_compute_H(walls,board,print_lines=True)


        mat = Visualise_single(A,B,C,x,colour_function=propagate_BEM_pressure,colour_function_args={"H":H,"scatterer":walls,"board":board},res=(40,40)).cpu().detach().numpy()
        # plt.imshow(mat,cmap='hot')
        # plt.show()


        layer = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)

        im = None
        def traverse(layer):
                A[1] += steps
                B[1] += steps
                C[1] += steps
                mat = Visualise_single(A,B,C,x,colour_function=propagate_BEM_pressure,colour_function_args={"H":H,"scatterer":walls,"board":board},res=(40,40)).cpu().detach().numpy()
                im = ax.matshow(mat,cmap='hot',vmax=5000)
                ax.set_title("XZ planae, Y="+str(xz_dy+steps*layer))
                
        rot_animation = animation.FuncAnimation(fig, traverse, frames=np.arange(0, step_y), interval=500)
        rot_animation.save('Traverse_SIM_XZ.gif', dpi=80, writer='imagemagick')



