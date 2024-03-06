if __name__ == '__main__':
    import pickle, sys, torch
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    from acoustools.Visualiser import Visualise_single
    from acoustools.BEM import load_multiple_scatterers, get_cache_or_compute_H, propagate_BEM_pressure
    from acoustools.Mesh import scatterer_file_name
    from acoustools.Utilities import TRANSDUCERS


    loss,results = pickle.load(open('Media/SavedResults/SphereLev.pth','rb'))

    fig = plt.figure()
    loss_ax = fig.add_subplot(1,2,1)
    img_ax = fig.add_subplot(1,2,2)


    loss = [l.cpu().detach().numpy() for l in loss]
    loss_ax.plot(loss)
    loss_ax.set_ylabel('Loss')
    loss_ax.set_xlabel('Iteration')

    loss_ax.spines['right'].set_color('none')
    loss_ax.spines['top'].set_color('none')


    wall_paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.175/2,0.175/2],rotys=[90,-90]) #Make mesh at 0,0,0
    walls.scale((1,19/12,19/12),reset=True,origin =False)
    walls.filename = scatterer_file_name(walls)
    
    H = get_cache_or_compute_H(walls, TRANSDUCERS)

    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    start = list(results.keys())[0]
    im = Visualise_single(A,B,C,results[start],propagate_BEM_pressure,colour_function_args={"H":H,"scatterer":walls,"board":TRANSDUCERS})

    img_ax.matshow(im.cpu().detach(),cmap='hot',vmax=9000) 
    
    def traverse(index):
        im = Visualise_single(A,B,C,results[index],propagate_BEM_pressure,colour_function_args={"H":H,"scatterer":walls,"board":TRANSDUCERS})
        img_ax.matshow(im.cpu().detach(),cmap='hot',vmax=9000) 
        img_ax.set_title("XZ plane, Step="+str(index+1))

        loss_ax.clear()
        loss_ax.plot(loss)
        loss_ax.set_ylabel('Loss')
        loss_ax.set_xlabel('Iteration')
        loss_ax.spines['bottom'].set_position('zero')
        loss_ax.plot([index,index],[0,loss[index].item()])
    

    rot_animation = animation.FuncAnimation(fig, traverse, frames=results.keys(), interval=500)
    rot_animation.save('Results.gif', dpi=80, writer='imagemagick')
    
