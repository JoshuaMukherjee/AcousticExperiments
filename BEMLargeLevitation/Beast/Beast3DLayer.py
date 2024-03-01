if __name__ == '__main__':
    from acoustools.read_beast_file import read_beast_file

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fileName = r"C:\Users\joshu\Documents\Figs\BEMLargeLevitation\Beast28-2-24\Scan DATE_28 2 2024; TIME_13 26.csv"

    mode = "Peak"             # Mode can be "RMS" or "Peak"
    amp_setting=1.0/1000     # amp_setting is how many volts per pascal, native 0dB gain on amp is 1mv/Pa for lowest noise.
    mic_gain_correction=0.81  # mic_gain_correction is gain due to incident angle on microphone, default is for 90 degrees. See below for reference values extracted from datasheet graph


    min_x = 63
    min_y = 163
    min_z = 65

    step_x = 24
    step_y = 24
    step_z = 12

    amp, phase = read_beast_file(fileName, mode, amp_setting, mic_gain_correction)
    amp_filtered = amp[min_x:min_x+step_x+1, min_y:min_y+step_y+1, min_z:min_z+step_z+1]
    amp_filtered = amp_filtered

    THRESHOLD = 0
    amp_filtered[amp_filtered < THRESHOLD] = np.nan

    x = np.linspace(min_x, min_x+step_x, step_x+1)
    y = np.linspace(min_y, min_y+step_y, step_y+1)
    z = np.linspace(min_z, min_z+step_z, step_z+1)
    x,y,z = np.meshgrid(x,y,z)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    layer = 0

    def traverse(layer):
            im = ax.matshow(amp_filtered[:,:,layer].T,cmap='hot',vmax=5000) #z=0 is the bottom
            ax.set_title("XY planae, Z="+str(layer))
            
    rot_animation = animation.FuncAnimation(fig, traverse, frames=np.arange(0, step_z), interval=500)
    rot_animation.save('Traverse.gif', dpi=80, writer='imagemagick')


