import pickle, torch

import matplotlib.pyplot as plt

N = [1,2,3,4,5,6,8,10,12,16,32]
M = 100

results = pickle.load(open('walls_quant.pth', 'rb'))

pressures = results['pressure']
forces = results['force']
gorkov = results['gorkov']

meths = ['wgs','gspat','bem']
labs = ['WGS:PM', 'GS-PAT:PM','WGS:BEM']

plt.subplot(3,2,1)
pressure_plot = {}
for meth in meths:
    pressure_plot[meth] = []
    
    for ps_n in pressures[meth]:
        means = []
        for p in pressures[meth][ps_n]:
            mean = torch.mean(p)
            means.append(mean)
        pressure_plot[meth].append((sum(means)/M).cpu().detach())


for i,meth in enumerate(meths):
    plt.plot(N, pressure_plot[meth][:-1], label=labs[i])
    plt.scatter(N, pressure_plot[meth][:-1], marker='.')
plt.legend()
plt.ylabel('Pressure (Pa)')

plt.subplot(3,2,3)
gorkov_plot = {}
for meth in meths:
    gorkov_plot[meth] = []
    
    for ps_n in gorkov[meth]:
        means = []
        for p in gorkov[meth][ps_n]:
            mean = torch.mean(p)
            means.append(mean)
        gorkov_plot[meth].append((sum(means)/M).cpu().detach())


for i,meth in enumerate(meths):
    plt.plot(N, gorkov_plot[meth][:-1], label=labs[i])
    plt.scatter(N, gorkov_plot[meth][:-1], marker='.')
plt.legend()
plt.xlabel('Points')
plt.ylabel('$U$')


force_plot_x = {}
force_plot_y = {}
force_plot_z = {}
for meth in meths:
    force_plot_x[meth] = []
    force_plot_y[meth] = []
    force_plot_z[meth] = []
    
    for ps_n in forces[meth]:
        means = []
        for p in forces[meth][ps_n]:
            p = p.squeeze()
            if len(p.shape) == 1:
                p = p.unsqueeze(0).unsqueeze(2)
            mean = torch.mean(p,dim=(0,2))
            means.append(mean)
        force_plot_x[meth].append((sum(means)/M)[0].cpu().detach())
        force_plot_y[meth].append((sum(means)/M)[1].cpu().detach())
        force_plot_z[meth].append((sum(means)/M)[2].cpu().detach())


plt.subplot(3,2,2)
for i,meth in enumerate(meths):
    plt.plot(N, [torch.abs(i) for i in force_plot_x[meth][:-1]], label=labs[i])
    plt.scatter(N, [torch.abs(i) for i in force_plot_x[meth][:-1]],marker='.')
plt.legend()
plt.ylabel('$|F_x|$ (N)')
plt.yscale('log')

plt.subplot(3,2,4)
for i,meth in enumerate(meths):
    plt.plot(N, [torch.abs(i) for i in force_plot_y[meth][:-1]], label=labs[i])
    plt.scatter(N, [torch.abs(i) for i in force_plot_y[meth][:-1]],marker='.')
plt.legend()
plt.ylabel('$|F_y|$ (N)')
plt.yscale('log')

plt.subplot(3,2,6)
for i,meth in enumerate(meths):
    plt.plot(N, [torch.abs(i) for i in force_plot_z[meth][:-1]], label=labs[i])
    plt.scatter(N, [torch.abs(i) for i in force_plot_z[meth][:-1]],marker='.')
plt.legend()
plt.xlabel('Points')
plt.ylabel('$|F_z|$ (N)')
plt.yscale('log')

plt.show()