if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig, device, TOP_BOARD, propagate_abs
    from acoustools.Solvers import iterative_backpropagation
    from acoustools.Visualiser import Visualise

    import torch, time

    board = TOP_BOARD

    p = create_points(1,1,x=0,y=0,z=0)
    x = iterative_backpropagation(p, board=board)

    A = torch.tensor((-0.09,0, 0.09)).to(device)
    B = torch.tensor((0.09,0, 0.09)).to(device)
    C = torch.tensor((-0.09,0, -0.09)).to(device)
    
    res = 200
    start = time.time_ns()
    Visualise(A,B,C, x, res=(res,res), points=p, show=True, colour_functions=[propagate_abs,], colour_function_args=[{"board":board}])
    end = time.time_ns()
    print((end-start)/1e9)