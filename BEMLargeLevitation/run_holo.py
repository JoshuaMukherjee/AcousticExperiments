from acoustools.Levitator import LevitatorController
import pickle

lev = LevitatorController(ids=(73,53))

# x = pickle.load(open(r'./BEMLargeLevitation\Paths\toTry\1\holo_stable_up.pth','rb'))
x = pickle.load(open('BEMLargeLevitation/Paths/holo.pth','rb'))

lev.levitate(x, permute=False)

input()

lev.turn_off()
lev.disconnect()

