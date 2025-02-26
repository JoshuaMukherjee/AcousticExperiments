import pickle, os

folder = './BEMLargeLevitation/Times/'

types = ['GPU/', 'CPU/', 'Mac/']

for t in types:
    print(t)
    pth = folder+t
    files = os.listdir(pth)
    for f in files:
        ts =pickle.load(open(pth+f,'rb'))
        start = ts[0]
        for t in ts:
            print((t-start) / 1e9)