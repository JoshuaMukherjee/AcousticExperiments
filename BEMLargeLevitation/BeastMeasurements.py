from acoustools.read_beast_file import read_beast_file
import matplotlib.pyplot as plt
import numpy as np

fileName = r"C:\Users\joshu\Documents\Figs\BEMLargeLevitation\Beast28-2-24\Scan DATE_28 2 2024; TIME_11 36.csv" #xz plane -bk 2. pg 17
# fileName = r"C:\Users\joshu\Documents\Figs\BEMLargeLevitation\Beast28-2-24\Scan DATE_28 2 2024; TIME_12 6.csv" #xy plane
# fileName = r"C:\Users\joshu\Documents\Figs\BEMLargeLevitation\Beast28-2-24\Scan DATE_28 2 2024; TIME_12 53.csv" #yz plane

mode = "Peak"             # Mode can be "RMS" or "Peak"
amp_setting=1.0/1000     # amp_setting is how many volts per pascal, native 0dB gain on amp is 1mv/Pa for lowest noise.
mic_gain_correction=0.81  # mic_gain_correction is gain due to incident angle on microphone, default is for 90 degrees. See below for reference values extracted from datasheet graph


min_x = 55
min_y = 175
min_z = 57

step_x = 40
step_y = 0
step_z = 40

amp, phase = read_beast_file(fileName, mode, amp_setting, mic_gain_correction)
amp_filtered = amp[min_x:min_x+step_x+1, min_y:min_y+step_y+1, min_z:min_z+step_z+1]

amp_squeeze = amp_filtered.squeeze()
amp_squeeze = amp_squeeze.T

plt.matshow(amp_squeeze,cmap='hot')
plt.colorbar()
plt.show()

