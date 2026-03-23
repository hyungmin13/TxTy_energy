#%%
import jax
import jax.numpy as jnp
from jax import random
from time import time
import optax
from jax import value_and_grad
from functools import partial
from jax import jit
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.interpolate import interpn
import h5py
from scipy.io import loadmat
import argparse
from Tecplot_mesh import tecplot_Mesh
from tqdm import tqdm
#%%
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        self.c=c

class PINN(PINNbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "network1":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        all_params["network1"] = self.c.network1.init_params(**self.c.network1_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        #all_params = self.c.problem.constraints(all_params)
        #valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network1.network_fn
        return all_params, model_fn, train_data

class PINN2(PINNbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "network1":{}, "network2":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        all_params["network1"] = self.c.network1.init_params(**self.c.network1_init_kwargs)
        all_params["network2"] = self.c.network2.init_params(**self.c.network2_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        #all_params = self.c.problem.constraints(all_params)
        #valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network1.network_fn
        model_fn2 = c.network2.network_fn2
        return all_params, model_fn, model_fn2, train_data



def equ_func(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
    return out, out_t

def equ_func2(all_params, g_batch, cotangent1, cotangent2, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent1, ))[1]
    out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent2,))
    return out_x, out_xx

def equ_func3(all_params, g_batch, cotangent1, cotangent2, cotangent3, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)                                                                                     
    def u_tt(batch):                                                                                                                
        return jax.jvp(u_t,(batch,), (cotangent1, ))[1]                                                                         
    def u_ttt(batch):                                                                                                               
        return jax.jvp(u_tt,(batch,),(cotangent2, ))[1]                                                                         
    out_xx, out_xxx = jax.jvp(u_ttt, (g_batch,), (cotangent3,))                                                                 
    return out_xx, out_xxx

def Derivatives(dynamic_params, all_params, g_batch, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']

    all_params["network1"]["layers"] = dynamic_params

    out_xx, out_xxx = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_xy, out_xxy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_xz, out_xxz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    
    _, out_xyy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_yy, out_yyy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_yz, out_yyz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    
    _, out_xzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), model_fns)
    _, out_yzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), model_fns)
    out_zz, out_zzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), model_fns)
    
    out_x, out_xt = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_y, out_yt = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_z, out_zt = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out, out_t = equ_func(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    
    uvwp = jnp.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)

    uts = jnp.concatenate([out_t[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
    uxs = jnp.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
    uys = jnp.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uzs = jnp.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uxts = jnp.concatenate([out_xt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
    uyts = jnp.concatenate([out_yt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
    uzts = jnp.concatenate([out_zt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
    uxxs = jnp.concatenate([out_xx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
    uxys = jnp.concatenate([out_xy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uxzs = jnp.concatenate([out_xz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uyys = jnp.concatenate([out_yy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uyzs = jnp.concatenate([out_yz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uzzs = jnp.concatenate([out_zz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uxxxs = jnp.concatenate([out_xxx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
    uxxys = jnp.concatenate([out_xxy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uxxzs = jnp.concatenate([out_xxz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uxyys = jnp.concatenate([out_xyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uyyys = jnp.concatenate([out_yyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uyyzs = jnp.concatenate([out_yyz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uxzzs = jnp.concatenate([out_xzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uyzzs = jnp.concatenate([out_yzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uzzzs = jnp.concatenate([out_zzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    deriv_mat = np.concatenate([np.expand_dims(uxs,2),np.expand_dims(uys,2),np.expand_dims(uzs,2)],2)

    Q = 0.5 * sum(-np.abs(0.5 * (deriv_mat[:, i, j] + deriv_mat[:, j, i]))**2 +
                  np.abs(0.5 * (deriv_mat[:, i, j] - deriv_mat[:, j, i]))**2 
                  for i in range(3) for j in range(3))

    Ty = uyts[:,2:3]-uzts[:,1:2] + uvwp[:,0:1]*(uxys[:,2:3] - uxzs[:,1:2]) + uvwp[:,1:2]*(uyys[:,2:3] - uyzs[:,1:2]) + uvwp[:,2:3]*(uyzs[:,2:3] - uzzs[:,1:2]) \
        - (uys[:,2:3] - uzs[:,1:2])*uxs[:,0:1] - (uzs[:,0:1] - uxs[:,2:3])*uys[:,0:1] - (uxs[:,1:2] - uys[:,0:1])*uzs[:,0:1] \
        - all_params["data"]["viscosity"]*(uxxys[:,2:3] + uyyys[:,2:3] + uyzzs[:,2:3] - uxxzs[:,1:2] - uyyzs[:,1:2] - uzzzs[:,1:2])

    Tx = (uys[:,2:3] - uzs[:,1:2])*uxs[:,1:2] + (uzs[:,0:1] - uxs[:,2:3])*uys[:,1:2] + (uxs[:,1:2] - uys[:,0:1])*uzs[:,1:2] \
        + all_params["data"]["viscosity"]*(uxxzs[:,0:1] + uyyzs[:,0:1] + uzzzs[:,0:1] - uxxxs[:,2:3] - uxyys[:,2:3] - uxzzs[:,2:3]) \
        - uzts[:,0:1] + uxts[:,2:3] - uvwp[:,0:1]*(uxzs[:,0:1] - uxxs[:,2:3]) - uvwp[:,1:2]*(uyzs[:,0:1] - uxys[:,2:3]) - uvwp[:,2:3]*(uzzs[:,0:1] - uxzs[:,2:3])


    return uvwp, uxs, uys, uzs, Tx, Ty, Q

def Derivatives_sep(dynamic_params, dynamic_params2, all_params, g_batch, model_fns, model_fns2):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref', 'T_ref']

    all_params["network1"]["layers"] = dynamic_params
    all_params["network2"]["layers"] = dynamic_params2

    T = model_fns2(all_params, g_batch).reshape(-1,1)*all_params["data"][keys[-1]]


    
    out_xx, out_xxx = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_xy, out_xxy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_xz, out_xxz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    
    _, out_xyy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_yy, out_yyy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_yz, out_yyz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    
    _, out_xzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), model_fns)
    _, out_yzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), model_fns)
    out_zz, out_zzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                    jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), model_fns)
    
    out_x, out_xt = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_y, out_yt = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_z, out_zt = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out, out_t = equ_func(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    
    uvwp = jnp.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)

    uts = jnp.concatenate([out_t[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
    uxs = jnp.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
    uys = jnp.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uzs = jnp.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uxts = jnp.concatenate([out_xt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
    uyts = jnp.concatenate([out_yt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
    uzts = jnp.concatenate([out_zt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
    uxxs = jnp.concatenate([out_xx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
    uxys = jnp.concatenate([out_xy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uxzs = jnp.concatenate([out_xz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uyys = jnp.concatenate([out_yy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uyzs = jnp.concatenate([out_yz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uzzs = jnp.concatenate([out_zz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uxxxs = jnp.concatenate([out_xxx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
    uxxys = jnp.concatenate([out_xxy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uxxzs = jnp.concatenate([out_xxz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uxyys = jnp.concatenate([out_xyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uyyys = jnp.concatenate([out_yyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
    uyyzs = jnp.concatenate([out_yyz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uxzzs = jnp.concatenate([out_xzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uyzzs = jnp.concatenate([out_yzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    uzzzs = jnp.concatenate([out_zzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
    deriv_mat = np.concatenate([np.expand_dims(uxs,2),np.expand_dims(uys,2),np.expand_dims(uzs,2)],2)

    Q = 0.5 * sum(-np.abs(0.5 * (deriv_mat[:, i, j] + deriv_mat[:, j, i]))**2 +
                  np.abs(0.5 * (deriv_mat[:, i, j] - deriv_mat[:, j, i]))**2 
                  for i in range(3) for j in range(3))

    Ty = uyts[:,2:3]-uzts[:,1:2] + uvwp[:,0:1]*(uxys[:,2:3] - uxzs[:,1:2]) + uvwp[:,1:2]*(uyys[:,2:3] - uyzs[:,1:2]) + uvwp[:,2:3]*(uyzs[:,2:3] - uzzs[:,1:2]) \
        - (uys[:,2:3] - uzs[:,1:2])*uxs[:,0:1] - (uzs[:,0:1] - uxs[:,2:3])*uys[:,0:1] - (uxs[:,1:2] - uys[:,0:1])*uzs[:,0:1] \
        - all_params["data"]["viscosity"]*(uxxys[:,2:3] + uyyys[:,2:3] + uyzzs[:,2:3] - uxxzs[:,1:2] - uyyzs[:,1:2] - uzzzs[:,1:2])

    Tx = (uys[:,2:3] - uzs[:,1:2])*uxs[:,1:2] + (uzs[:,0:1] - uxs[:,2:3])*uys[:,1:2] + (uxs[:,1:2] - uys[:,0:1])*uzs[:,1:2] \
        + all_params["data"]["viscosity"]*(uxxzs[:,0:1] + uyyzs[:,0:1] + uzzzs[:,0:1] - uxxxs[:,2:3] - uxyys[:,2:3] - uxzzs[:,2:3]) \
        - uzts[:,0:1] + uxts[:,2:3] - uvwp[:,0:1]*(uxzs[:,0:1] - uxxs[:,2:3]) - uvwp[:,1:2]*(uyzs[:,0:1] - uxys[:,2:3]) - uvwp[:,2:3]*(uzzs[:,0:1] - uxzs[:,2:3])

    return uvwp, uxs, uys, uzs, Tx, Ty, Q

def Derivatives_vec(dynamic_params, dynamic_params2, all_params, g_batch, model_fns, model_fns2):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref', 'T_ref']
    all_params["network1"]["layers"] = dynamic_params
    all_params["network2"]["layers"] = dynamic_params2
    out_x, out_tx = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_y, out_ty = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_z, out_tz = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_xx, out_xxx = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_xy, out_xyy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_xz, out_xzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_xxz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_xxy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_yy, out_yyy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_yz, out_yyz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_yzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)
    out_zz, out_zzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)
    out_T = model_fns2(all_params, g_batch)
    p = out_T[:,0:1]*1.185*all_params["data"][keys[-2]]
    T = out_T[:,1:2]*all_params["data"][keys[-1]]
    u = out_y[:,2:3] - out_z[:,1:2]
    ut = out_ty[:,2:3] - out_tz[:,1:2]
    ux = (out_xy[:,2:3] - out_xz[:,1:2])/all_params["domain"]["domain_range"]["x"][1]
    uy = (out_yy[:,2:3] - out_yz[:,1:2])/all_params["domain"]["domain_range"]["y"][1]
    uz = (out_yz[:,2:3] - out_zz[:,1:2])/all_params["domain"]["domain_range"]["z"][1]
    uxx = (out_xxy[:,2:3] - out_xxz[:,1:2])/all_params["domain"]["domain_range"]["x"][1]**2
    uyy = (out_yyy[:,2:3] - out_yyz[:,1:2])/all_params["domain"]["domain_range"]["y"][1]**2
    uzz = (out_yzz[:,2:3] - out_zzz[:,1:2])/all_params["domain"]["domain_range"]["z"][1]**2
    v = out_z[:,0:1] - out_x[:,2:3]
    vt = out_tz[:,0:1] - out_tx[:,2:3]
    vx = out_xz[:,0:1] - out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
    vy = out_yz[:,0:1] - out_xy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
    vz = out_zz[:,0:1] - out_xz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
    vxx = out_xxz[:,0:1] - out_xxx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
    vyy = out_yyz[:,0:1] - out_xyy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
    vzz = out_zzz[:,0:1] - out_xzz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
    w = out_x[:,1:2] - out_y[:,0:1]
    wt = out_tx[:,1:2] - out_ty[:,0:1]
    wx = out_xx[:,1:2] - out_xy[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
    wy = out_xy[:,1:2] - out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
    wz = out_xz[:,1:2] - out_yz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
    wxx = out_xxx[:,1:2] - out_xxy[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
    wyy = out_xyy[:,1:2] - out_yyy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
    wzz = out_xzz[:,1:2] - out_yzz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
    uvw = np.concatenate([u,v,w,p,T],1)
    uts = np.concatenate([ut,vt,wt],1)
    uxs = np.concatenate([ux,vx,wx],1)
    uys = np.concatenate([uy,vy,wy],1)
    uzs = np.concatenate([uz,vz,wz],1)
    uxxs = np.concatenate([uxx, vxx, wxx, uyy, vyy, wyy, uzz, vzz, wzz],1)
    deriv_mat = np.concatenate([np.expand_dims(uxs,2), np.expand_dims(uys,2), np.expand_dims(uzs,2)],2)
    #Q = uy*vx + uz*wx +vz*wy - 0.5*(uy**2+uz**2+vx**2+vz**2+wx**2+wy**2)
    Q = 0.5 * sum(-np.abs(0.5 * (deriv_mat[:, i, j] + deriv_mat[:, j, i]))**2 +
                  np.abs(0.5 * (deriv_mat[:, i, j] - deriv_mat[:, j, i]))**2
                  for i in range(3) for j in range(3))
    return uvw, uxs, uys, uzs, T, T, Q

def Tecplotfile_gen(c, path, name, all_params, domain_range, output_shape, order, timestep, is_ground, is_mean, model_fn, model_fn2=None):
    
    # Load the parameters
    pos_ref = all_params["domain"]["in_max"].flatten()
    dynamic_params = all_params["network1"].pop("layers")
    if model_fn2:
        dynamic_params2 = all_params["network2"].pop("layers")
    # Create the evaluation grid
    gridbase = [np.linspace(domain_range[key][0], domain_range[key][1], output_shape[i]) for i, key in enumerate(['t', 'x', 'y', 'z'])]
    print(gridbase)
    gridbase_n = [gridbase[i].copy()/pos_ref[i] for i in range(len(gridbase))]
    if order[0] == 0:
        if order[1] == 1:
            z_e, y_e, x_e = np.meshgrid(gridbase[-3], gridbase[-2], gridbase[-1], indexing='ij')
            z_n, y_n, x_n = np.meshgrid(gridbase_n[-3], gridbase_n[-2], gridbase_n[-1], indexing='ij')
        else:
            z_e, x_e, y_e = np.meshgrid(gridbase[-3], gridbase[-1], gridbase[-2], indexing='ij')
            z_n, x_n, y_n = np.meshgrid(gridbase_n[-3], gridbase_n[-1], gridbase_n[-2], indexing='ij')
    elif order[0] == 1:
        if order[1] == 0:
            y_e, z_e, x_e = np.meshgrid(gridbase[-2], gridbase[-3], gridbase[-1], indexing='ij')
            y_n, z_n, x_n = np.meshgrid(gridbase_n[-2], gridbase_n[-3], gridbase_n[-1], indexing='ij')
        else:
            y_e, x_e, z_e = np.meshgrid(gridbase[-2], gridbase[-1], gridbase[-3], indexing='ij')
            y_n, x_n, z_n = np.meshgrid(gridbase_n[-2], gridbase_n[-1], gridbase_n[-3], indexing='ij')
    elif order[0] == 2:
        if order[1] == 0:
            x_e, z_e, y_e = np.meshgrid(gridbase[-1], gridbase[-3], gridbase[-2], indexing='ij')
            x_n, z_n, y_n = np.meshgrid(gridbase_n[-1], gridbase_n[-3], gridbase_n[-2], indexing='ij')
        else:
            x_e, y_e, z_e = np.meshgrid(gridbase[-1], gridbase[-2], gridbase[-3], indexing='ij')
            x_n, y_n, z_n = np.meshgrid(gridbase_n[-1], gridbase_n[-2], gridbase_n[-3], indexing='ij')   
    t_e = np.zeros(output_shape[1:]) + gridbase[0][timestep]
    t_n = np.zeros(output_shape[1:]) + gridbase_n[0][timestep]
    eval_grid = np.concatenate([t_n.reshape(-1,1), x_n.reshape(-1,1), y_n.reshape(-1,1), z_n.reshape(-1,1)], axis=1)
    eval_grid_e = np.concatenate([t_e.reshape(-1,1), x_e.reshape(-1,1), y_e.reshape(-1,1), z_e.reshape(-1,1)], axis=1)
    print(np.max(eval_grid[:,0]), np.max(eval_grid[:,1]), np.max(eval_grid[:,2]), np.max(eval_grid[:,3]))
    # Load Ground truth data if is_ground is True
    if is_ground:
        ground_data = np.load(path + 'ground/ts_' + str(timestep).zfill(2) + '.npy')
    if is_mean:
        mean_data = np.load(path + 'mean')

    # Evaluate the derivatives
    if "network2" in all_params.keys():
        if "equation2" in c.equation_init_kwargs.keys():
            print('network2 and equation2')
            uvwp, uxs, uys, uzs, Tx, Ty, Q = zip(*[Derivatives_sep(dynamic_params, dynamic_params2, all_params, eval_grid[i:i+10000], model_fn, model_fn2)
                                                for i in range(0, eval_grid.shape[0], 10000)])
        else:
            print('network2')
            uvwp, uxs, uys, uzs, Tx, Ty, Q = zip(*[Derivatives_vec(dynamic_params, dynamic_params2, all_params, eval_grid[i:i+10000], model_fn, model_fn2)
                                                for i in range(0, eval_grid.shape[0], 10000)])
    else:
        uvwp, uxs, uys, uzs, Tx, Ty, Q = zip(*[Derivatives(dynamic_params, all_params, eval_grid[i:i+10000], model_fn)
                                            for i in range(0, eval_grid.shape[0], 10000)])
    
    # Concatenate the results
    uvwp = np.concatenate(uvwp, axis=0)
    uxs = np.concatenate(uxs, axis=0)
    uys = np.concatenate(uys, axis=0)
    uzs = np.concatenate(uzs, axis=0)
    Q = np.concatenate(Q, axis=0)
    Tx = np.concatenate(Tx, axis=0)
    Ty = np.concatenate(Ty, axis=0)
    #uvwp[:,3] = uvwp[:,3] - np.mean(uvwp[:,3])

    if is_ground:
        grounds = [ground_data[:,i+4].reshape(output_shape[1:]) for i in range(3)]
        errors = [np.sqrt(np.square(uvwp[:,i].reshape(output_shape[1:]) - grounds[i])) for i in range(3)]
        if ground_data.shape[1] > 7:
            p_ground = ground_data[:,7].reshape(output_shape[1:])
            p_error = np.sqrt(np.square(uvwp[:,3].reshape(output_shape[1:]) - p_ground))
        if ground_data.shape[1] > 8:
            temp_ground = ground_data[:,8].reshape(output_shape[1:])
            temp_error = np.sqrt(np.square(uvwp[:,4].reshape(output_shape[1:]) - temp_ground))
    if is_mean:
        means = [mean_data['vel'][:,i].reshape(output_shape[1:]) for i in range(3)]
        flucs = [uvwp[:,i].reshape(output_shape[1:]) - means[i] for i in range(3)]

    # Tecplot file generation
    filename = path + 'Tecplotfile/' + name + '/ts_' + str(timestep) + '.dat'
    if os.path.isdir(path + 'Tecplotfile/' + name):
        pass
    else:
        os.mkdir(path + 'Tecplotfile/' + name)
    X, Y, Z = output_shape[1:]
    vars = [('u_pred[m/s]',np.float32(uvwp[:,0].reshape(-1))), ('v_pred[m/s]',np.float32(uvwp[:,1].reshape(-1))), 
            ('w_pred[m/s]',np.float32(uvwp[:,2].reshape(-1))), ('p_pred[Pa]',np.float32(uvwp[:,3].reshape(-1))),
            ('Q[1/s^2]', np.float32(Q.reshape(-1)))]
    if is_ground:
        vars += [('u_error[m/s]', np.float32(errors[0].reshape(-1))),
                 ('v_error[m/s]', np.float32(errors[1].reshape(-1))),
                 ('w_error[m/s]', np.float32(errors[2].reshape(-1)))]
        if ground_data.shape[1] > 7:
            vars += [('p_error[Pa]', np.float32(p_error.reshape(-1)))]
        if ground_data.shape[1] > 8:
            vars += [('temp_error[K]', np.float32(temp_error.reshape(-1)))]
    if is_mean:
        vars += [('u_fluc[m/s]', np.float32(flucs[0].reshape(-1))),
                 ('v_fluc[m/s]', np.float32(flucs[1].reshape(-1))),
                 ('w_fluc[m/s]', np.float32(flucs[2].reshape(-1)))]
    pad = 27
    tecplot_Mesh(filename, X, Y, Z, x_e.reshape(-1), y_e.reshape(-1), z_e.reshape(-1), vars, pad)

    if os.path.isdir(path + 'npyresult/' + name):
        pass
    else:
        print('check')
        os.mkdir(path + 'npyresult/' + name)
    np.save(path + 'npyresult/' + name + f'/ts_{timestep:02d}' + '.npy', np.concatenate([eval_grid_e, uvwp, uxs, uys, uzs, Tx, Ty], axis=1))
#%%
if __name__ == "__main__":
    from domain import *
    from trackdata import *
    from network import *
    from constants import *
    from problem import *
    from txt_reader import *
    import os
    parser = argparse.ArgumentParser(description='PINN')
    parser.add_argument('-f', '--foldername', type=str, help='foldername', default='HIT')
    parser.add_argument('-c', '--config', type=str, help='configuration', default='eval_config')
    args = parser.parse_args()

    # Get evaluation configuration
    cur_dir = os.getcwd()
    config_txt = cur_dir + '/' + args.config + '.txt'
    data = parse_tree_structured_txt(config_txt)

    # Get model constants
    with open(os.path.dirname(cur_dir)+ '/' + data['path'] + args.foldername +'/summary/constants.pickle','rb') as f:
        constants = pickle.load(f)
    values = list(constants.values())
    print(values)
    if values[4]:
        c = Constants(run = values[0],
                    domain_init_kwargs = values[1],
                    data_init_kwargs = values[2],
                    network1_init_kwargs = values[3],
                    network2_init_kwargs = values[4],
                    problem_init_kwargs = values[5],
                    optimization_init_kwargs = values[6],
                    equation_init_kwargs = values[7],)
    else:
        c = Constants(run = values[0],
                    domain_init_kwargs = values[1],
                    data_init_kwargs = values[2],
                    network1_init_kwargs = values[3],
                    problem_init_kwargs = values[5],
                    optimization_init_kwargs = values[6],
                    equation_init_kwargs = values[7],)
    if values[4]:
        run = PINN2(c)
    else:
        run = PINN(c)

    # Get model parameters
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if values[4]:
        checkpoint_list2 = sorted(glob(run.c.model_out_dir2+'/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    with open(checkpoint_list[-1],"rb") as f:
        model_params = pickle.load(f)
    if values[4]:
        with open(checkpoint_list2[-1],"rb") as f:
            model_params2 = pickle.load(f)
    if values[4]:
        all_params, model_fn, model_fn2, train_data = run.test()
    else:
        all_params, model_fn, train_data = run.test()
    model = Model(all_params["network1"]["layers"], model_fn)
    if values[4]:
        model2 = Model(all_params["network2"]["layers"], model_fn2)
    all_params["network1"]["layers"] = from_state_dict(model, model_params).params
    if values[4]:
        all_params["network2"]["layers"] = from_state_dict(model2, model_params2).params
    domain_range = data['tecplot_init_kwargs']['domain_range']
    output_shape = data['tecplot_init_kwargs']['out_shape']
    order = data['tecplot_init_kwargs']['order']
    timesteps = data['tecplot_init_kwargs']['timestep']
    is_ground = data['tecplot_init_kwargs']['is_ground']
    path = data['tecplot_init_kwargs']['path']
    is_mean = data['tecplot_init_kwargs']['is_mean']
    path = os.path.dirname(cur_dir) + '/' + path
    pos_ref = all_params["domain"]["in_max"].flatten()
    if "network2" in all_params.keys():
        for timestep in timesteps:
            Tecplotfile_gen(c, path, args.foldername, all_params, domain_range, output_shape, order, timestep, is_ground, is_mean, model_fn, model_fn2)

    else:
        for timestep in timesteps:
            Tecplotfile_gen(c, path, args.foldername, all_params, domain_range, output_shape, order, timestep, is_ground, is_mean, model_fn)