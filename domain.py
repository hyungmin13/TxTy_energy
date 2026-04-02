#%%
import numpy as np
import jax.numpy as jnp
import os
from glob import glob
class Domainbase:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError

    @staticmethod
    def bound_sampler(all_params,grids):
        raise NotImplementedError
    
    @staticmethod
    def normalize(all_params, grids):
        raise NotImplementedError

    @staticmethod
    def sampler(all_params):
        raise NotImplementedError

    
class Domain(Domainbase):
    @staticmethod
    def init_params(**kwargs):
        domain_params = {}
        for key, value in kwargs.items():
            domain_params[key] = value
        return domain_params

    @staticmethod
    def bound_sampler(all_params, grids):
        bound_keys = all_params["domain"]["bound_keys"]
        arg_keys = ['t', 'x', 'y', 'z']
        total_bound = {bound_keys[i]:{arg_keys[j]:[] for j in range(len(arg_keys))} for i in range(len(bound_keys))}
        for i, bound_key in enumerate(bound_keys):
            for j in range(len(arg_keys)):
                total_bound[bound_key][arg_keys[j]] = grids['eqns'][arg_keys[j]]
            for arg_key in arg_keys:
                if arg_key in bound_key:
                    if 'u' in bound_key:
                        total_bound[bound_key][arg_key] = np.array([grids['eqns'][arg_key][-1]])
                    else:
                        total_bound[bound_key][arg_key] = np.array([grids['eqns'][arg_key][0]])
            if 'ic' in bound_key:
                total_bound[bound_key]['t'] = grids['eqns']['t'][0]
        grids.update(total_bound)
        return grids
    
    @staticmethod
    def normalize(all_params, grids):
        domain_range = all_params["domain"]["domain_range"]
        key_list = list(grids.keys())
        arg_keys = ['t', 'x', 'y', 'z']
        for i in range(len(key_list)):
            for arg_key in arg_keys:
                grids[key_list[i]][arg_key] = grids[key_list[i]][arg_key]/domain_range[arg_key][1]
            #grids[key_list[i]]['t'] = grids[key_list[i]]['t']*domain_range[arg_key][1]*frequency
        return grids
    @staticmethod
    def stretched_to_uniform_1d(H, yc, n_inner, n_outer, method="tanh", beta=3.0, a=20.0):
        eta_inner = np.linspace(0.0, 1.0, n_inner)

        if method == "tanh":
            y_inner = yc * (1.0 - np.tanh(beta * (1.0 - eta_inner)) / np.tanh(beta))
        elif method == "algebraic":
            y_inner = yc * (a**eta_inner - 1.0) / (a - 1.0)
        else:
            raise ValueError("method must be 'tanh' or 'algebraic'")

        y_outer = np.linspace(yc, H, n_outer)
        y = np.concatenate([y_inner[:-1], y_outer])

        return y
    @staticmethod
    def symmetric_stretched_uniform_1d(H, yc, n_inner, n_center, method="tanh", beta=3.0, a=20.0):

        eta = np.linspace(0.0, 1.0, n_inner)

        if method == "tanh":
            y_bottom = yc * (1.0 - np.tanh(beta * (1.0 - eta)) / np.tanh(beta))
        elif method == "algebraic":
            y_bottom = yc * (a**eta - 1.0) / (a - 1.0)
        else:
            raise ValueError("method must be 'tanh' or 'algebraic'")

        y_center = np.linspace(yc, H - yc, n_center)
        y_top = H - y_bottom[::-1]

        y = np.concatenate([
            y_bottom[:-1],
            y_center[:-1],
            y_top
        ])
        return y

    @staticmethod
    def sampler(all_params):
        domain_range = all_params["domain"]["domain_range"]
        grid_size = all_params["domain"]["grid_size"]
        path = all_params["data"]['path']
        try:
            fine_boundary = all_params["domain"]["fine_boundary"]
        except:
            print('fine_boundary is not defined, using uniform grid')
        cur_dir = os.getcwd()
        filenames = sorted(glob(os.path.dirname(cur_dir)+path+'*.npy'))
        t = []
        for filename in filenames:
            temp = np.load(filename)
            t.append(temp[0,0])
        t = np.array(t)
        arg_keys = ['t', 'x', 'y', 'z']
        grids = {'eqns':{arg_keys[j]:[] for j in range(len(arg_keys))}}
        grids['eqns']['t'] = t

        for i, arg_key in enumerate(arg_keys):
            bound_key = [b for b in all_params["domain"]["bound_keys"] if arg_key in b]
            if len(bound_key)==1:
                grids['eqns'][arg_key] = Domain.stretched_to_uniform_1d(domain_range[arg_key][1],
                                                                        fine_boundary[arg_key],
                                                                        int(grid_size[i]*0.2),
                                                                        grid_size[i]-int(grid_size[i]*0.2),
                                                                        method=all_params["domain"]['method'])
            elif len(bound_key)==2:
                grids['eqns'][arg_key] = Domain.symmetric_stretched_uniform_1d(domain_range[arg_key][1],
                                                                        fine_boundary[arg_key],
                                                                        int(grid_size[i]*0.2),
                                                                        grid_size[i]-int(grid_size[i]*0.2),
                                                                        method=all_params["domain"]['method'])    
            else:
                grids['eqns'][arg_key] = np.linspace(domain_range[arg_key][0], domain_range[arg_key][1], grid_size[i])

        grids = Domain.bound_sampler(all_params, grids)
        grids = Domain.normalize(all_params, grids)
        all_params["domain"]["in_min"] = jnp.array([[domain_range['t'][0], domain_range['x'][0], domain_range['y'][0], domain_range['z'][0]]])
        all_params["domain"]["in_max"] = jnp.array([[domain_range['t'][1], domain_range['x'][1], domain_range['y'][1], domain_range['z'][1]]])
        return grids, all_params
    


if __name__ == "__main__":
    from trackdata import *
    all_params = {"domain":{}, "data":{}}
    path = '/RBC_G8_DNS/npdata/lv6_xbound/'
    data_keys = ['pos', 'vel', 'T']
    viscosity = 15*10**(-6)
    all_params["data"] = Data.init_params(path = path, data_keys = data_keys, viscosity = viscosity)
    
    domain_range = {'t':(0,7.4), 'x':(0,0.2), 'y':(0,0.2), 'z':(0,0.2)}
    grid_size = [51, 200, 200, 200]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']
    fine_boundary = {'x':0.01, 'y':0.01, 'z':0.01}
    method = "tanh"
    all_params["domain"] = Domain.init_params(domain_range = domain_range, grid_size = grid_size, bound_keys = bound_keys,
                                              fine_boundary = fine_boundary, method = method)
    
    grids, all_params = Domain.sampler(all_params)

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    mesh_xyz = np.meshgrid(grids['eqns']['x'], grids['eqns']['y'], grids['eqns']['z'], indexing='ij')
    
    print(grids['eqns']['t'])

    fig, axes = plt.subplots(figsize=(8,8), nrows=2, ncols=2)
    for i in range(3):
        j, k = (i//2, i%2)
        im = axes[j,k].imshow(mesh_xyz[i][0,:,:],cmap='jet')
        divider = make_axes_locatable(axes[j,k])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.show()
# %%
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(grids['eqns']['x'])),grids['eqns']['x']*0.2, marker='o', ms=3, label="tanh + uniform")
    plt.xlabel("Grid index")
    plt.ylabel("y")
    plt.title("1D mesh")
    plt.grid(True)
    plt.legend()
    plt.show()

# %%
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(grids['eqns']['x']))[:-1],np.diff(grids['eqns']['x'])*0.2, marker='o', ms=3, label="tanh + uniform")
    plt.xlabel("Grid index")
    plt.ylabel("y")
    plt.title("1D mesh")
    plt.grid(True)
    plt.legend()
    plt.show()
# %%
    print(200**3*50/10000)
# %%
    print(np.diff(grids['eqns']['x'])*0.2)
# %%
