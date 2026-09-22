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
from soap_jax import soap
from scipy.spatial import KDTree
import itertools

def RAD_probability(residual_score, k = 1.0, c = 1.0, eps = 1e-12,):
    residual_score = jnp.asarray(residual_score)
    residual_score = jnp.nan_to_num(residual_score,nan=0.0,posinf=0.0,neginf=0.0,)
    residual_score = jnp.maximum(residual_score, 0.0)
    weighted_residual = residual_score**k
    rad_weight = (weighted_residual/ (jnp.mean(weighted_residual) + eps)+ c)
    probability = rad_weight / (jnp.sum(rad_weight) + eps)
    return probability

def RAD_sampling(key, candidate_points, residual_score, num_residual_points, k = 1.0, c = 1.0, replace = False):
    probability = RAD_probability(residual_score,k=k,c=c,)
    print(candidate_points.shape)
    print(num_residual_points)
    selected_points = random.choice(key,candidate_points,shape=(num_residual_points,),replace=replace,p=probability,)
    #selected_points = candidate_points[selected_indices]
    return selected_points, probability


class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

@partial(jax.jit, static_argnums=(1, 2, 5, 10))
def PINN_update(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, grids, particles, particle_vel, particle_scale, particle_bd, model_fn):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(equation_fn, argnums=0)(dynamic_params, all_params, grids, particles, particle_vel, particle_scale, particle_bd, model_fn)
    updates, model_states = optimiser_fn(grads, model_states, dynamic_params)
    dynamic_params = optax.apply_updates(dynamic_params, updates)
    return lossval, model_states, dynamic_params

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c

class PINN(PINNbase):
    def train(self):
        all_params = {"domain":{}, "data":{}, "network1":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network1"] = self.c.network1.init_params(**self.c.network1_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        
        # Initialize optmiser
        learn_rate = optax.exponential_decay(self.c.optimization_init_kwargs["learning_rate"],
                                             self.c.optimization_init_kwargs["decay_step"],
                                             self.c.optimization_init_kwargs["decay_rate"],)
        optimiser = self.c.optimization_init_kwargs["optimiser"](learning_rate=learn_rate, b1=0.95, b2=0.95,
                                                                 weight_decay=0.01, precondition_frequency=5)
        #optimiser = self.c.optimization_init_kwargs["optimiser"](learning_rate=learn_rate)
        model_states = optimiser.init(all_params["network1"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = c.network1.network_fn
        dynamic_params = all_params["network1"].pop("layers")

        # Define equation function
        equation1_fn = self.c.equation1.Loss
        report_fn = self.c.equation1.Loss_report
        residual_fn = self.c.equation1.residual_score
        # Input data and grids
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        train_data, idx = self.c.data.weight_balance_coeff(all_params, train_data)
        if 'path_w' in all_params['data'].keys():
            print('wall data detected')
            wall_data = self.c.data.wall_data(all_params.copy())
        valid_data = self.c.problem.exact_solution(all_params.copy())

        # Input key initialization
        key, batch_key = random.split(key)
        num_keysplit = 10
        keys = random.split(batch_key, num = num_keysplit)
        keys_split = [random.split(keys[i], num = self.c.optimization_init_kwargs["n_steps"]) for i in range(num_keysplit)]
        keys_iter = [iter(keys_split[i]) for i in range(num_keysplit)]
        keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
        # Static parameters
        leaves, treedef = jax.tree_util.tree_flatten(all_params)
        static_params = tuple(x if isinstance(x,(np.ndarray, jnp.ndarray)) else None for x in leaves)
        static_leaves = tuple(None if isinstance(x,(np.ndarray, jnp.ndarray)) else x for x in leaves)
        static_keys = (static_leaves, treedef)
        print(np.unique(train_data['pos'][:,0]))
        # Initializing batches
        print(np.max(train_data['pos'][:,-1]), train_data['pos'].shape)
        print(np.max(valid_data['pos'][:,-1]), valid_data['pos'].shape)
        N_p = train_data['pos'].shape[0]
        perm_p = random.permutation(keys_next[0], N_p)
        data_p = []
        data_v = []
        data_s = []
        for i in range(N_p//self.c.optimization_init_kwargs["p_batch"]):
            batch_p = train_data['pos'][perm_p[i*self.c.optimization_init_kwargs["p_batch"]:(i+1)*self.c.optimization_init_kwargs["p_batch"]],:]
            batch_v = train_data['vel'][perm_p[i*self.c.optimization_init_kwargs["p_batch"]:(i+1)*self.c.optimization_init_kwargs["p_batch"]],:]
            batch_s = train_data['scale'][perm_p[i*self.c.optimization_init_kwargs["p_batch"]:(i+1)*self.c.optimization_init_kwargs["p_batch"]],:]
            data_p.append(batch_p)
            data_v.append(batch_v)
            data_s.append(batch_s)
        data_p.append(train_data['pos'][perm_p[-1-self.c.optimization_init_kwargs["p_batch"]:-1],:])
        data_v.append(train_data['vel'][perm_p[-1-self.c.optimization_init_kwargs["p_batch"]:-1],:])
        data_s.append(train_data['scale'][perm_p[-1-self.c.optimization_init_kwargs["p_batch"]:-1],:])
        p_batches = itertools.cycle(data_p)
        v_batches = itertools.cycle(data_v)
        s_batches = itertools.cycle(data_s)
        p_batch = next(p_batches)
        v_batch = next(v_batches)
        s_batch = next(s_batches)
        g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                           grids['eqns'][arg], 
                                           shape=(self.c.optimization_init_kwargs["p_batch"],)) 
                             for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
        print(np.unique(g_batch[:,0]))
        b_batches = []
        for b_key in all_params["domain"]["bound_keys"]:
            b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                            grids[b_key][arg], 
                                            shape=(self.c.optimization_init_kwargs["b_batch"],)) 
                                            for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            b_batches.append(b_batch)
        
        RAD_collocation_points = None
        # Initializing the update function
        update = PINN_update.lower(model_states, optimiser_fn, equation1_fn, dynamic_params, static_params, static_keys, g_batch, p_batch, v_batch, s_batch, b_batches, model_fn).compile()

            # Training loop
        RAD_check = 0
        RAD = False
        n = int(all_params['domain']['max_RAD'])
        for i in range(self.c.optimization_init_kwargs["n_steps"]):
            keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
            p_batch = next(p_batches)
            v_batch = next(v_batches)
            s_batch = next(s_batches)
            #p_batch = random.choice(keys_next[0],train_data['pos'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            #v_batch = random.choice(keys_next[0],train_data['vel'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            if not RAD:
                g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                            grids['eqns'][arg], 
                                            shape=(self.c.optimization_init_kwargs["p_batch"],)) 
                                for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            else:
                g_uni_batch = jnp.stack([random.choice(keys_next[k+1], grids['eqns'][arg], shape=(self.c.optimization_init_kwargs["e_batch"],)) for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
                g_rad_batch = random.choice(keys_next[0],RAD_collocation_points,shape=(self.c.optimization_init_kwargs["p_batch"]-self.c.optimization_init_kwargs["e_batch"],))
                g_batch = jnp.concatenate([g_uni_batch,g_rad_batch],0)        
            b_batches = []
            for b_key in all_params["domain"]["bound_keys"]:
                b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                                grids[b_key][arg], 
                                                shape=(self.c.optimization_init_kwargs["b_batch"],)) 
                                    for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
                b_batches.append(b_batch)
            lossval, model_states, dynamic_params = update(model_states, dynamic_params, static_params, g_batch, p_batch, v_batch, s_batch, b_batches)
            if all_params["domain"]["RAD_sampling"]:
                if i%all_params['domain']['RAD_interval'] == 0:
                    residual_score = residual_fn(dynamic_params, all_params, g_batch, p_batch, v_batch, s_batch, b_batches, model_fn) 
                    new_rad_points, prob = RAD_sampling(keys_next[-1], g_batch, residual_score, all_params["domain"]["RAD_num"], k = all_params["domain"]["k"], c = all_params["domain"]["c"])
                    if RAD_collocation_points is not None:
                        RAD_collocation_points = jnp.concatenate([RAD_collocation_points, new_rad_points],0)
                        print(RAD_collocation_points.shape)
                        if RAD_collocation_points.shape[0]>all_params["domain"]["max_RAD"]:
                            np.save(self.c.report_out_dir + 'RAD_points'+str(RAD_check)+'.npy',RAD_collocation_points[:n//2,:])
                            RAD_collocation_points = RAD_collocation_points[n//2:,:]
                            RAD_check = RAD_check + 1
                        if RAD_collocation_points.shape[0]>n//5:
                            RAD = True
                            print('RAD_activated')
                     
                    else:
                        RAD_collocation_points = new_rad_points
            
            self.report(i, report_fn, dynamic_params, all_params, p_batch, v_batch, s_batch, g_batch, b_batches, valid_data, keys_iter[-1], self.c.optimization_init_kwargs["save_step"], model_fn)
            self.save_model(i, dynamic_params, all_params, self.c.optimization_init_kwargs["save_step"], model_fn)

    def save_model(self, i, dynamic_params, all_params, save_step, model_fns):
        model_save = (i % save_step == 0)
        if model_save:
            all_params["network1"]["layers"] = dynamic_params
            model = Model(all_params["network1"]["layers"], model_fns)
            serialised_model = to_state_dict(model)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
        return

    def report(self, i, report_fn, dynamic_params, all_params, p_batch, v_batch, s_batch, g_batch, b_batch, valid_data, e_batch_key, save_step, model_fns, Tx_batch = None, Ty_batch = None):    
        save_report = (i % save_step == 0)
        if save_report:
            all_params["network1"]["layers"] = dynamic_params
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            if 'T' in valid_data.keys():
                e_batch_T = random.choice(e_key, valid_data['T'], shape = (self.c.optimization_init_kwargs["e_batch"],))
                if 'Tx' in valid_data.keys():
                    Losses = report_fn(dynamic_params, all_params, g_batch, p_batch, v_batch, s_batch, e_batch_pos, e_batch_vel, b_batch, model_fns, particle_Tx = Tx_batch, particle_Ty = Ty_batch, e_batch_T = e_batch_T)
                else:
                    Losses = report_fn(dynamic_params, all_params, g_batch, p_batch, v_batch, s_batch, e_batch_pos, e_batch_vel, b_batch, model_fns, e_batch_T = e_batch_T)
            else:
                if 'Tx' in valid_data.keys():
                    Losses = report_fn(dynamic_params, all_params, g_batch, p_batch, v_batch, s_batch, e_batch_pos, e_batch_vel, b_batch, model_fns, particle_Tx = Tx_batch, particle_Ty = Ty_batch)
                else:
                    Losses = report_fn(dynamic_params, all_params, g_batch, p_batch, v_batch, s_batch, e_batch_pos, e_batch_vel, b_batch, model_fns)

            print(f"step_num : {i:<{12}} total_loss : {Losses[0]:<{12}.{5}} u_loss : {Losses[1]:<{12}.{5}} "
                    f"v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} con_loss : {Losses[4]:<{12}.{5}} "
                    f"NS1_loss : {Losses[5]:<{12}.{5}} NS2_loss : {Losses[6]:<{12}.{5}} NS3_loss : {Losses[7]:<{12}.{5}} Eng_loss : {Losses[8]:<{12}.{5}} "
                    f"Tbu_loss : {Losses[9]:<{12}.{5}} Tbb_loss : {Losses[10]:<{11}.{5}} Tx_loss : {Losses[11]:<{12}.{5}} Ty_loss : {Losses[12]:<{12}.{5}}"
                    f"u_error : {Losses[13]:<{12}.{5}} v_error : {Losses[14]:<{12}.{5}} w_error : {Losses[15]:<{12}.{5}} T_error : {Losses[16]:<{12}.{5}}")
            with open(self.c.report_out_dir + "reports.txt", "a") as f:
                f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} "
                        f"{Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {Losses[7]:<{12}.{5}} {Losses[8]:<{12}.{5}} {Losses[13]:<{12}.{5}} {Losses[14]:<{12}.{5}} {Losses[15]:<{12}.{5}} {Losses[16]:<{12}.{5}}\n")
            f.close()
        return

#%%
if __name__=="__main__":
    from domain import *
    from trackdata import *
    from network import *
    from constants import *
    from problem import *
    from equation import *
    from txt_reader import *
    import argparse
    
    parser = argparse.ArgumentParser(description='TBL_PINN')
    parser.add_argument('-n', '--name', type=str, help='run name', default='HIT_k1')
    parser.add_argument('-c', '--config', type=str, help='configuration', default='test_txt')
    args = parser.parse_args()
    cur_dir = os.getcwd()
    input_txt = cur_dir + '/' + args.config + '.txt' 
    data = parse_tree_structured_txt(input_txt)
    c = Constants(**data)

    run = PINN(c)
    run.train()
#%%
