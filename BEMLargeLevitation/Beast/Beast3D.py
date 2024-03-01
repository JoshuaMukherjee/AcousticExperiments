if __name__ == '__main__':
    from acoustools.read_beast_file import read_beast_file

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fileName = r"C:\Users\joshu\Documents\Figs\BEMLargeLevitation\Beast28-2-24\Scan DATE_28 2 2024; TIME_13 26.csv"

    mode = "Peak"             # Mode can be "RMS" or "Peak"
    amp_setting=1.0/1000     # amp_setting is how many volts per pascal, native 0dB gain on amp is 1mv/Pa for lowest noise.
    mic_gain_correction=0.81  # mic_gain_correction is gain due to incident angle on microphone, default is for 90 degrees. See below for reference values extracted from datasheet graph

    

    def plt_sphere(c, r):
        
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)

        ax.plot_surface(x+c[0], y+c[1], z+c[2])


    min_x = 63
    min_y = 163
    min_z = 65

    step_x = 24
    step_y = 24
    step_z = 12

    amp, phase = read_beast_file(fileName, mode, amp_setting, mic_gain_correction)
    amp_filtered = amp[min_x:min_x+step_x+1, min_y:min_y+step_y+1, min_z:min_z+step_z+1]

    THRESHOLD = 1500
    amp_filtered[amp_filtered < THRESHOLD] = np.nan

    x = np.linspace(min_x, min_x+step_x, step_x+1)
    y = np.linspace(min_y, min_y+step_y, step_y+1)
    z = np.linspace(min_z, min_z+step_z, step_z+1)
    x,y,z = np.meshgrid(x,y,z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    img = ax.scatter(x, y, z, c=amp_filtered, cmap='hot')
    # ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect([1,1,1/2])
    # plt_sphere([76,175,77],10)
    fig.colorbar(img)


    SHOW = True

    if SHOW:
        plt.show()
    
    else:

        def rotate(angle):
            ax.view_init(azim=angle)

        print("Making animation")
        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
        rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')

    # print(amp_filtered)

