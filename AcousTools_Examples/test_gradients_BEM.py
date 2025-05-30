
if __name__ == "__main__":
    from acoustools.Utilities import forward_model_batched, forward_model_grad, forward_model_second_derivative_unmixed, forward_model_second_derivative_mixed, TRANSDUCERS, create_points, add_lev_sig, propagate,DTYPE, transducers
    from acoustools.Solvers import wgs
    from acoustools.Gorkov import get_finite_diff_points_all_axis
    from acoustools.Utilities import device, propagate_abs
    import acoustools.Constants as c
    from acoustools.Mesh import load_scatterer, get_centres_as_points, scale_to_diameter, get_centre_of_mass_as_points
    from acoustools.BEM import compute_E, BEM_forward_model_grad, BEM_forward_model_second_derivative_mixed, BEM_forward_model_second_derivative_unmixed, propagate_BEM_pressure

    from acoustools.Visualiser import Visualise, ABC
    import torch, vedo

    # torch.random.manual_seed(18)

    torch.set_printoptions(linewidth=400)


    board = transducers(16)


    def return_mixed_points(points,stepsize=0.000135156253, stepsize_x=None,stepsize_y=None,stepsize_z=None ):
        '''
        Only works for N=1
        '''
        if points.shape[2] > 1:
            raise RuntimeError("Only for N=1")
        
        if stepsize_x is None:
            stepsize_x = stepsize
        
        if stepsize_y is None:
            stepsize_y = stepsize

        if stepsize_z is None:
            stepsize_z = stepsize


        mixed_points = points.clone().repeat((1,1,13))
        #Set x's
        mixed_points[:,0,1] += stepsize_x
        mixed_points[:,0,2] += stepsize_x
        mixed_points[:,0,3] -= stepsize_x
        mixed_points[:,0,4] -= stepsize_x
        mixed_points[:,0,5] += stepsize_x
        mixed_points[:,0,6] += stepsize_x
        mixed_points[:,0,7] -= stepsize_x
        mixed_points[:,0,8] -= stepsize_x
        #Set y's
        mixed_points[:,1,1] += stepsize_y
        mixed_points[:,1,2] -= stepsize_y
        mixed_points[:,1,3] += stepsize_y
        mixed_points[:,1,4] -= stepsize_y
        mixed_points[:,1,9] += stepsize_y
        mixed_points[:,1,10] += stepsize_y
        mixed_points[:,1,11] -= stepsize_y
        mixed_points[:,1,12] -= stepsize_y
        #Set z's
        mixed_points[:,2,5] += stepsize_z
        mixed_points[:,2,6] -= stepsize_z
        mixed_points[:,2,7] += stepsize_z
        mixed_points[:,2,8] -= stepsize_z
        mixed_points[:,2,9] += stepsize_z
        mixed_points[:,2,10] -= stepsize_z
        mixed_points[:,2,11] += stepsize_z
        mixed_points[:,2,12] -= stepsize_z


        return mixed_points

    path = "../BEMMedia"
    sphere_pth =  path+"/Sphere-lam2.stl"
    sphere = load_scatterer(sphere_pth, dy=-0.06, dz=-0.04) #Make mesh at 0,0,0
    print(get_centre_of_mass_as_points(sphere))
    scale_to_diameter(sphere, 0.03)
    centres = get_centres_as_points(sphere)



    N=1
    B=1
    D=3
    for i in range(1):
        # points = create_points(N,B,x=0.02, y=-0.005, z=-0.04)
        points = create_points(N,B, z=0.06)
        print(points)
        # points = torch.autograd.Variable(points.data, requires_grad=True).to(device).to(DTYPE)

        E,F,G,H = compute_E(sphere, centres, board, return_components=True, path=path)


        activations = wgs(centres, A=E).to(DTYPE)
        
        # Visualise(*ABC(0.1), activations,points=points, colour_functions=[propagate_BEM_pressure], colour_function_args=[{"board":board,"path":path,"H":H,"scatterer":sphere}])
        # exit()
        # activations = add_lev_sig(activations, mode='Twin')

        
        Ex, Ey, Ez= BEM_forward_model_grad(points, sphere, board, path=path, H=H)
        Exx, Eyy, Ezz = BEM_forward_model_second_derivative_unmixed(points, sphere.clone(), board, H=H, path=path)
        Exy, Exz, Eyz = BEM_forward_model_second_derivative_mixed(points, sphere.clone(), board, H=H, path=path)


        # stepsize = 0.000135156253
        stepsize = c.wavelength / 16
        # stepsize = 1e-4
        print(stepsize)

        fin_diff_points = get_finite_diff_points_all_axis(points,stepsize=stepsize)
        E_fd = compute_E(sphere, fin_diff_points, board, path=path, H=H)
        pressure_points = propagate(activations, fin_diff_points, A=E_fd)
        pressure = pressure_points[:,:N]
        pressure_fin_diff = pressure_points[:,N:]
        # split = torch.reshape(pressure_fin_diff,(B,2, ((2*D))*N // 2))
        split = torch.reshape(pressure_fin_diff,(B,2, -1))
        grad = (split[:,0,:] - split[:,1,:]) / (2*stepsize)

        # p  = torch.abs(F@activations)
        # P_a = torch.autograd.grad (p, points, retain_graph=True, create_graph=True)[0]   # creates graph of first derivative
        # pa = torch.abs(P_a)
        # p_angle = torch.angle(P_a)

        print("Grad","Analytical","Finite Differences","Ratio",sep="\t")
        Px  = torch.abs(Ex@activations)
        print("px", Px.item(), torch.abs(grad[0,0]).item(),torch.abs(grad[0,0]).item() / Px.item(),sep="\t")
        Py  = torch.abs(Ey@activations)
        print("py", Py.item(),torch.abs(grad[0,1]).item(),torch.abs(grad[0,1]).item() / Py.item(),sep="\t")
        Pz  = torch.abs(Ez@activations)
        print("pz", Pz.item(), torch.abs(grad[0,2]).item(),torch.abs(grad[0,2]).item() / Pz.item(),sep="\t")
        
        print()


        print("grad", 'Analytical', 'Finite Differences', 'Ratio', sep='\t')

        grad_unmixed = (split[:,0,:] - 2*pressure + split[:,1,:]) / (stepsize**2)

        Pxx = torch.abs(Exx@activations)
        print("Pxx", Pxx.item(), torch.abs(grad_unmixed[0,0]).item(), torch.abs(grad_unmixed[0,0]).item() / Pxx.item(),sep="\t")
        Pyy = torch.abs(Eyy@activations)
        print("Pyy", Pyy.item(), torch.abs(grad_unmixed[0,1]).item(), torch.abs(grad_unmixed[0,1]).item() / Pyy.item(),sep="\t")
        Pzz = torch.abs(Ezz@activations)
        print("Pzz", Pzz.item(), torch.abs(grad_unmixed[0,2]).item(), torch.abs(grad_unmixed[0,2]).item() / Pzz.item(),sep="\t")

        print()

        stepsize_x = stepsize 
        stepsize_y = stepsize 
        stepsize_z = stepsize 

        mixed_points = return_mixed_points(points,stepsize_x=stepsize_x, stepsize_y=stepsize_y, stepsize_z=stepsize_z)
        E_fd_m = compute_E(sphere, mixed_points, board, path=path, H=H)
        mixed_pressure_points = propagate(activations, mixed_points, A=E_fd_m)
              
        Pxy = torch.abs(Exy@activations)
                
        mixed_pressure_fin_diff_xy = mixed_pressure_points[:,1:5] * torch.tensor([1,-1,-1,1])
        Pxy_fd = torch.sum(mixed_pressure_fin_diff_xy) / (4*stepsize_x*stepsize_y)
        print("Pxy",Pxy.item(), torch.abs(Pxy_fd).item(),torch.abs(Pxy_fd).item()/Pxy.item(),sep='\t')

        Pxz = torch.abs(Exz@activations)
        mixed_pressure_fin_diff_xz = mixed_pressure_points[:,5:9] * torch.tensor([1,-1,-1,1])
        Pxz_fd = torch.sum(mixed_pressure_fin_diff_xz) / (4*stepsize_x*stepsize_z)
        print("Pxz",Pxz.item(), torch.abs(Pxz_fd).item(), torch.abs(Pxz_fd).item()/Pxz.item() ,sep='\t')

        Pyz = torch.abs(Eyz@activations)
        mixed_pressure_fin_diff_yz = mixed_pressure_points[:,9:] * torch.tensor([1,-1,-1,1])
        Pyz_fd = torch.sum(mixed_pressure_fin_diff_yz) / (4*stepsize_y*stepsize_z)
        print("Pyz",Pyz.item(), torch.abs(Pyz_fd).item(), torch.abs(Pyz_fd).item()/Pyz.item() ,sep='\t')


