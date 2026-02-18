if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig, device
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise

    import torch, time

    p = create_points(1,1,x=0,y=0,z=0)
    x = wgs(p)
    x = add_lev_sig(x, mode='Twin')

    A = torch.tensor((-0.01,0, 0.01)).to(device)
    B = torch.tensor((0.01,0, 0.01)).to(device)
    C = torch.tensor((-0.01,0, -0.01)).to(device)
    
    res = 200
    start = time.time_ns()
    Visualise(A,B,C, x, res=(res,res), points=p, show=True)
    end = time.time_ns()
    print((end-start)/1e9)