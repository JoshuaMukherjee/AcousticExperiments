import pickle
import matplotlib.pyplot as plt

loss,result =  pickle.load(open('Media/SavedResults/SphereLev.pth','rb'))

losses_clean = []
for l in loss:
    losses_clean.append(l.detach().cpu().item())

plt.plot(losses_clean)
plt.show()

