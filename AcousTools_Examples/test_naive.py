if __name__ == '__main__':
    from acoustools.Solvers import naive
    from acoustools.Utilities import create_points, propagate_abs

    p = create_points(2,1)
    print(p)
    x = naive(p)
    print(x.shape)
    print(propagate_abs(x,p))
    
    p = p.squeeze(0)
    print(p)
    x = naive(p)
    print(x.shape)
    print(propagate_abs(x,p))

