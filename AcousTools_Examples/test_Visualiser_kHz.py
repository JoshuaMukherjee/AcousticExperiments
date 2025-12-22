if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig, device, propagate_abs
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise, ABC
    from acoustools.Constants import c_0, pi

    k10 = 2*pi / (c_0 / 10000)
    k20 = 2*pi / (c_0 / 20000)
    k80 = 2*pi / (c_0 / 80000)
    k120 = 2*pi / (c_0 / 120000)
    k1000 = 2*pi / (c_0 / 1000000)
    

    p = create_points(1,1,0,0,0)
    
    x10 = wgs(p, k=k10)
    x20 = wgs(p, k=k20)
    x40 = wgs(p)
    x80 = wgs(p, k=k80)
    x120 = wgs(p, k=k120)
    x1000 = wgs(p, k=k1000)
    # x = add_lev_sig(x)
  
    res = 300
    Visualise(*ABC(0.1), [x10, x20, x40, x80, x120, x1000], arangement=(2,3), link_ax=None,
              res=(res,res), points=p, show=True, colour_functions=[propagate_abs, propagate_abs, propagate_abs, propagate_abs,propagate_abs, propagate_abs], 
              colour_function_args=[{"k":k10}, {"k":k20}, {}, {"k":k80}, {"k":k120}, {"k":k1000}])
