from acoustools.Levitator import LevitatorController
import pickle

lev = LevitatorController(ids=(73,53))

x = pickle.load(open('./BEMLargeLevitation/Paths/holo.pth','rb'))

lev.levitate(x)

input()

lev.turn_off()
lev.disconnect()