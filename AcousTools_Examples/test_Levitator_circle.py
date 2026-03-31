if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    from acoustools.Export.Holo import save_holograms, load_holograms
    import math, time

    COMPUTE = False
    N = 250
    if COMPUTE:
        print('Computing...')
        xs = []
        poss=[]
        
        radius = 0.02
        for i in range(N):
            t = ((3.1415926*2) / N) * i
            x = radius * math.sin(t)
            y = radius * math.cos(t)
            print(i,x,y,0)
            poss.append((x,y,0))
            p = create_points(1,1,x=x,y=y,z=0)
            x = wgs(p)
            x = add_lev_sig(x)
            xs.append(x)
        print('Finished\nConnecting...')
        save_holograms(xs,f'Circle-{N}.holo')
    else:
        xs = load_holograms(f'Circle-{N}.holo')

    lev = LevitatorController()

    lev.levitate(xs[0])
    input()
    print('Moving...')
    try:
        while True:
            for x in xs:
                lev.levitate(0.35*x)
                time.sleep(0.1)
    except:  
        print('Finished')
        input()
        lev.disconnect()



