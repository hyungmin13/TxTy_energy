#%%
import numpy as np
from glob import glob
import os
from scipy.interpolate import PchipInterpolator
import itertools
class Database:
    @staticmethod
    def init_parmas(path, s_range, t_range, track_limit):
        raise NotImplementedError
    @staticmethod
    def data_load(filename):
        raise NotImplementedError
    @staticmethod
    def track_filter(track):
        raise NotImplementedError
    @staticmethod
    def domain_filter(s_trange):
        raise NotImplementedError
    @staticmethod
    def data_split(data, ratio):
        raise NotImplementedError
    
class Data(Database):
    @staticmethod
    #def init_params(path, domain_range, timeskip, track_limit, frequency, data_keys, viscosity):
    def init_params(**kwargs):
        data_params = {}
        for key, value in kwargs.items():
            data_params[key] = value
        return data_params

    @staticmethod
    def data_load_npy(filename, data_keys):
        data = np.load(filename)

        cols = {'pos':4, 'vel':3, 'p':1, 'T':1, 'Tx':1, 'Ty':1, 'acc':3}
        required_keys = ['pos', 'vel']
        for required_key in required_keys:
            if required_key not in data_keys:
                raise ValueError(f"Required '{required_key}' key is not in data_keys.")
        all_data = {}
        idx = 0
        for col in cols.keys():
            if col in data_keys:
                all_data[col] = data[:,idx:idx+cols[col]]
                idx += cols[col]
        #print(data[0,:])
        #print(all_data['Tx'])
        #print(all_data['Ty'])
        return all_data

    @staticmethod
    def domain_filter(all_data_, data_keys, domain_range):
        index = np.where((all_data_['pos'][:,0]>=domain_range['t'][0])&(all_data_['pos'][:,0]<=domain_range['t'][1])&
                         (all_data_['pos'][:,1]>=domain_range['x'][0])&(all_data_['pos'][:,1]<=domain_range['x'][1])&
                         (all_data_['pos'][:,2]>=domain_range['y'][0])&(all_data_['pos'][:,2]<=domain_range['y'][1])&
                         (all_data_['pos'][:,3]>=domain_range['z'][0])&(all_data_['pos'][:,3]<=domain_range['z'][1]))
        all_data_ = {data_keys[i]:all_data_[data_keys[i]][index[0],:] for i in range(len(data_keys))}
        return all_data_
    
    @staticmethod
    def input_normalize(all_params, data):
        domain_range = all_params["domain"]["domain_range"]
        arg_keys = ['t', 'x', 'y', 'z']
        for i in range(data['pos'].shape[1]):
            data['pos'][:,i] = data['pos'][:,i]/domain_range[arg_keys[i]][1]
        return data

    @staticmethod
    def output_normalize(all_params, data):
        vel_ref = {}
        vel_ref['u_ref'] = all_params['data']['u_ref']
        vel_ref['v_ref'] = all_params['data']['v_ref']
        vel_ref['w_ref'] = all_params['data']['w_ref']
        vel_ref['p_ref'] = all_params['data']['p_ref']
        vel_ref['T_ref'] = all_params['data']['T_ref']
        all_params["data"].update(vel_ref)
        return all_params

    @staticmethod
    def train_data(all_params):
        cur_dir = os.getcwd()
        path = all_params["data"]["path"]
        domain_range = all_params["domain"]["domain_range"]
        data_keys = all_params["data"]["data_keys"]
        #bound_keys = all_params["data"]["bound_keys"]

        filenames = sorted(glob(os.path.dirname(cur_dir)+path+'*.npy'))
        print(filenames)
        datas = {data_keys[i]:[] for i in range(len(data_keys))}

        seed_number = np.arange(0,1000)
        np.random.seed(42)
        seeds = np.random.choice(seed_number, len(filenames))

        for t, filename in enumerate(filenames):
            all_data_ = Data.data_load_npy(filename, data_keys)
            for i in range(len(data_keys)): datas[data_keys[i]].append(all_data_[data_keys[i]])
        for j in range(len(data_keys)): datas[data_keys[j]] = np.concatenate(datas[data_keys[j]], 0, dtype=np.float64)
        datas = Data.domain_filter(datas, data_keys, domain_range)
        train_data = Data.input_normalize(all_params, datas)
        all_params = Data.output_normalize(all_params, train_data)
        all_params["data"]["in_mean"] = np.array([[np.mean(train_data['pos'][:,0]), np.mean(train_data['pos'][:,1]), np.mean(train_data['pos'][:,2]), np.mean(train_data['pos'][:,3])]])
        all_params["data"]["in_std"] = np.array([[np.std(train_data['pos'][:,0]), np.std(train_data['pos'][:,1]), np.std(train_data['pos'][:,2]), np.std(train_data['pos'][:,3])]])
        return train_data, all_params
    @staticmethod
    def weight_balance_coeff(all_params, train_data):
        def get_profile(data, idx, c):
            A_sort = data[idx]
            cumsum = np.concatenate([[0],np.cumsum(c)])
            A_profile = np.array([np.mean(np.abs(A_sort[cumsum[i]:cumsum[i+1]]))
                                for i in range(len(c))])
            return A_profile

        def get_local_amplitude(z, interp):
            z = np.asarray(z)
            d = np.minimum(z, 1.0 - z)
            return np.exp(interp(d))

        def get_scale(z,interp,profile,target_ratio=10.0):
            z = np.asarray(z)
            local = get_local_amplitude(z,interp)
            wall = 0.5*(profile[0]+profile[-1]) 
            #Should be sampled from profile, otherwise original_ratio become inf

            center = get_local_amplitude(0.5, interp)
            original_ratio = center / wall

            if original_ratio <= target_ratio:
                print('no_scaling')
                return np.ones_like(z)

            gamma = (np.log(target_ratio)/ np.log(original_ratio))
            beta = 1.0 - gamma
            return (center / local)**beta

        z = train_data['pos'][:, 3]

        c, bins = np.histogram(z,bins=100,range=(0.0, 1.0))

        bin_idx = (np.digitize(z, bins) - 1)
        bin_idx = np.clip(bin_idx,0,len(c) - 1)
        bin_idx_ = np.argsort(bin_idx)


        comps = ['u', 'v', 'w']

        profiles = {name: get_profile(train_data['vel'][:, i],bin_idx_,c)
            for i, name in enumerate(comps)}

        z_profile = get_profile(z, bin_idx_, c)


        # Remove empty / invalid bins
        valid = ((c > 0)& np.isfinite(z_profile))

        for name in comps:
            valid &= (np.isfinite(profiles[name])& (profiles[name] > 0))
        z_profile = z_profile[valid]
        for name in comps:
            profiles[name] = profiles[name][valid]

        # Full-domain interpolation
        log_raw = {name: PchipInterpolator(z_profile, np.log(profiles[name]))
                    for name in comps}

        z_half = z_profile[z_profile <= 0.5]
        log_half = {name: 0.5 * (log_raw[name](z_half) + log_raw[name](1.0 - z_half))
                    for name in comps}

        interps = {name: PchipInterpolator(z_half, log_half[name])
                    for name in comps}
        scale = [get_scale(z,interps[name],profiles[name],target_ratio=10.0).reshape(-1,1) for name in comps]
        train_data['scale'] = np.concatenate(scale,1)
        return train_data, bin_idx_

        

if __name__ == "__main__":
    from domain import *
    from pathlib import Path
    import matplotlib.pyplot as plt
    from jax import random
    all_params = {"data":{}, "domain":{}}

    cur_dir = os.getcwd()
    #path = '/RBC_G8_DNS/npdata/lv6_xbound/'
    path = '/RBC_G8_DNS/npdata/lv4_pc/'
    data_keys = ['pos', 'vel',]
    viscosity = 2.64565e-3

    domain_range = {'t':(0,7.5), 'x':(0,8), 'y':(0,8), 'z':(0,1)}
    grid_size = [51, 200, 200, 200]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']
    u_ref = 0.26
    v_ref = 0.26
    w_ref = 0.4
    p_ref = 0.26
    T_ref = 0.5
    all_params["data"] = Data.init_params(path = path, 
                                          data_keys = data_keys, 
                                          viscosity = viscosity,
                                          u_ref = u_ref,
                                          v_ref = v_ref,
                                          w_ref = w_ref,
                                          p_ref = p_ref,
                                          T_ref = T_ref)
    all_params["domain"] = Domain.init_params(domain_range = domain_range, 
                                              bound_keys = bound_keys,
                                              grid_size = grid_size)
    
    train_data, all_params = Data.train_data(all_params)
    train_data, idx = Data.weight_balance_coeff(all_params, train_data)
    global_key = random.PRNGKey(42)
    key, batch_key = random.split(global_key)
    num_keysplit = 10
    keys = random.split(batch_key, num = num_keysplit)
    keys_split = [random.split(keys[i], num = 1000000) for i in range(num_keysplit)]
    keys_iter = [iter(keys_split[i]) for i in range(num_keysplit)]
    keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
    N_p = train_data['pos'].shape[0]
    perm_p = random.permutation(keys_next[0], N_p)
    data_p = []
    data_v = []
    data_scale = []
    for i in range(N_p//10000):
        batch_p = train_data['pos'][perm_p[i*10000:(i+1)*10000],:]
        batch_v = train_data['vel'][perm_p[i*10000:(i+1)*10000],:]
        batch_scale = train_data['scale'][perm_p[i*10000:(i+1)*10000],:]
        data_p.append(batch_p)
        data_v.append(batch_v)
        data_scale.append(batch_scale)
    data_p.append(train_data['pos'][perm_p[-1-10000:-1],:])
    data_v.append(train_data['vel'][perm_p[-1-10000:-1],:])
    data_scale.append(train_data['scale'][perm_p[-1-10000:-1],:])
    p_batches = itertools.cycle(data_p)
    v_batches = itertools.cycle(data_v)
    s_batches = itertools.cycle(data_scale)
    p_batch = next(p_batches)
    v_batch = next(v_batches)
    s_batch = next(s_batches)
#%%
    print(np.max(np.abs(train_data['vel'][idx,0]*train_data['scale'][idx,0])),
          np.min(np.abs(train_data['vel'][idx,0]*train_data['scale'][idx,0])))
    print(np.max(np.abs(train_data['vel'][idx,1]*train_data['scale'][idx,1])),
          np.min(np.abs(train_data['vel'][idx,1]*train_data['scale'][idx,1])))
    print(np.max(np.abs(train_data['vel'][idx,2]*train_data['scale'][idx,2])),
          np.min(np.abs(train_data['vel'][idx,2]*train_data['scale'][idx,2])))
#%%
    
#%%
    save_dir = Path("./RBC_vel_results")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
        constrained_layout=True,
    )
    im0 = axes[0].plot(np.abs(train_data['vel'][idx,0]*train_data['scale'][idx,0])
    )
    im1 = axes[1].plot(np.abs(train_data['vel'][idx,1]*train_data['scale'][idx,1])
    )
    im2 = axes[2].plot(np.abs(train_data['vel'][idx,-1]*train_data['scale'][idx,-1])
    )
    fig.savefig(
        save_dir /
        f"profile.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
