import pickle, torch
import matplotlib.pyplot as plt


# mean_pressure = pickle.load(open('mean_pressure.pth','rb'))
# max_pressure = pickle.load(open('max_pressure.pth','rb'))
# min_pressure = pickle.load(open('min_pressure.pth','rb'))

# for k in mean_pressure:
#     set_p = mean_pressure[k]
#     plt.scatter(range(30),[p.cpu().detach() for p in set_p])

# plt.show()

imgs = pickle.load(open('imgs.pth','rb'))

per_step = {}


for set in imgs:
    for i in range(len(imgs[set])):
        if i not in per_step:
            per_step[i] = []
        else:
            per_step[i] += imgs[set][i].flatten().tolist()

labels, data = per_step.keys(), per_step.values()



plt.boxplot(data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.show()