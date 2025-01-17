import pickle
from acoustools.Levitator import LevitatorController

holos = pickle.load(open('Media/SavedResults/force_tests.pth','rb'))

i = 0


lev = LevitatorController(ids=(73))

disconnected = False

try:
    done = False
    while not done:
        print(i, -0.01 + 0.0005*i)
        lev.levitate(holos[i])
        input()
        i += 1
        if i > len(holos):
            done = True
    lev.disconnect()
    disconnected = True
except KeyboardInterrupt:
    pass
finally:
    if not disconnected: lev.disconnect()