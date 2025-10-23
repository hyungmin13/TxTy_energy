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

class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

@partial(jax.jit, static_argnums=(1, 2, 5, 12))
def PINN_update(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, grids, particles, particle_vel, particle_bd, particle_Tx, particle_Ty, model_fn):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(equation_fn, argnums=0)(dynamic_params, all_params, grids, particles, particle_vel, particle_bd, particle_Tx, particle_Ty, model_fn)
    updates, model_states = optimiser_fn(grads, model_states, dynamic_params)
    dynamic_params = optax.apply_updates(dynamic_params, updates)
    return lossval, model_states, dynamic_params

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c

class PINN(PINNbase):
    def train(self,numb=0,**kwargs):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        
        model_params = model_params
        # Initialize optmiser
        learn_rate = optax.exponential_decay(self.c.optimization_init_kwargs["learning_rate"],
                                             self.c.optimization_init_kwargs["decay_step"],
                                             self.c.optimization_init_kwargs["decay_rate"],)
        optimiser = self.c.optimization_init_kwargs["optimiser"](learning_rate=learn_rate, b1=0.95, b2=0.95,
                                                                 weight_decay=0.01, precondition_frequency=5)
        model_states = optimiser.init(all_params["network"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = c.network.network_fn
        dynamic_params = all_params["network"].pop("layers")

        # Define equation function
        equation_fn = self.c.equation.Loss
        report_fn = self.c.equation.Loss_report

        # Input data and grids
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        model = Model(all_params['network']['layers'], model_fn)
        all_params["network"]["layers"] = from_state_dict(model, model_params).params
        valid_data = self.c.problem.exact_solution(all_params.copy())
        #model_states = optimiser.init(all_params["network"]["layers"])
        #optimiser_fn = optimiser.update
        #model_fn = c.network.network_fn
        #dynamic_params = all_params["network"].pop("layers")
        print(all_params['data']['u_ref'])
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

        # Initializing batches
        p_batch = random.choice(keys_next[0],train_data['pos'],shape=(self.c.optimization_init_kwargs["p_batch"],))
        v_batch = random.choice(keys_next[0],train_data['vel'],shape=(self.c.optimization_init_kwargs["p_batch"],))
        Tx_batch = random.choice(keys_next[0],train_data['Tx'],shape=(self.c.optimization_init_kwargs["p_batch"],))
        Ty_batch = random.choice(keys_next[0],train_data['Ty'],shape=(self.c.optimization_init_kwargs["p_batch"],))
        g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                           grids['eqns'][arg], 
                                           shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                             for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
        b_batches = []
        for b_key in all_params["domain"]["bound_keys"]:
            b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                            grids[b_key][arg], 
                                            shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            b_batches.append(b_batch)

        # Initializing the update function
        update = PINN_update.lower(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, g_batch, p_batch, v_batch, b_batches, Tx_batch, Ty_batch, model_fn).compile()
        
        # Training loop
        for i in range(self.c.optimization_init_kwargs["n_steps"]):
            keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
            p_batch = random.choice(keys_next[0],train_data['pos'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            v_batch = random.choice(keys_next[0],train_data['vel'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            Tx_batch = random.choice(keys_next[0],train_data['Tx'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            Ty_batch = random.choice(keys_next[0],train_data['Ty'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                            grids['eqns'][arg], 
                                            shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            b_batches = []
            for b_key in all_params["domain"]["bound_keys"]:
                b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                                grids[b_key][arg], 
                                                shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                    for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
                b_batches.append(b_batch)
            lossval, model_states, dynamic_params = update(model_states, dynamic_params, static_params, g_batch, p_batch, v_batch, b_batches, Tx_batch, Ty_batch)
        
        
            self.report(numb+i, report_fn, dynamic_params, all_params, p_batch, v_batch, g_batch, b_batch, Tx_batch, Ty_batch, valid_data, keys_iter[-1], self.c.optimization_init_kwargs["save_step"], model_fn)
            self.save_model(numb+i, dynamic_params, all_params, self.c.optimization_init_kwargs["save_step"], model_fn)

    def save_model(self, i, dynamic_params, all_params, save_step, model_fns):
        model_save = (i % save_step == 0)
        if model_save:
            all_params["network"]["layers"] = dynamic_params
            model = Model(all_params["network"]["layers"], model_fns)
            serialised_model = to_state_dict(model)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
        return
    
    def report(self, i, report_fn, dynamic_params, all_params, p_batch, v_batch, g_batch, b_batch, Tx_batch, Ty_batch, valid_data, e_batch_key, save_step, model_fns):
        save_report = (i % save_step == 0)
        if save_report:
            all_params["network"]["layers"] = dynamic_params
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            if 'T' in valid_data.keys():
                e_batch_T = random.choice(e_key, valid_data['T'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            v_pred = model_fns(all_params, e_batch_pos)
            print(all_params["data"]['u_ref'])
            u_error = jnp.sqrt(jnp.mean((all_params["data"]["u_ref"]*v_pred[:,0:1] - e_batch_vel[:,0:1])**2)/jnp.mean(e_batch_vel[:,0:1]**2))
            v_error = jnp.sqrt(jnp.mean((all_params["data"]["v_ref"]*v_pred[:,1:2] - e_batch_vel[:,1:2])**2)/jnp.mean(e_batch_vel[:,1:2]**2))
            w_error = jnp.sqrt(jnp.mean((all_params["data"]["w_ref"]*v_pred[:,2:3] - e_batch_vel[:,2:3])**2)/jnp.mean(e_batch_vel[:,2:3]**2))
            if v_pred.shape[1] == 5:
                T_error = jnp.sqrt(jnp.mean((all_params["data"]["T_ref"]*v_pred[:,4] - e_batch_T)**2)/jnp.mean(e_batch_T**2))

            Losses = report_fn(dynamic_params, all_params, g_batch, p_batch, v_batch, b_batch, Tx_batch, Ty_batch, model_fns)
            if v_pred.shape[1] == 5:
                print(f"step_num : {i:<{12}} u_loss : {Losses[1]:<{12}.{5}} v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} u_error : {u_error:<{12}.{5}} v_error : {v_error:<{12}.{5}} w_error : {w_error:<{12}.{5}} T_error : {T_error:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} {Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {Losses[7]:<{12}.{5}} {Losses[8]:<{12}.{5}} {u_error:<{12}.{5}} {v_error:<{12}.{5}} {w_error:<{12}.{5}} {T_error:<{12}.{5}}\n")
            else:
                print(f"step_num : {i:<{12}} u_loss : {Losses[1]:<{12}.{5}} v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} u_error : {u_error:<{12}.{5}} v_error : {v_error:<{12}.{5}} w_error : {w_error:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} {Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {Losses[7]:<{12}.{5}} {0.0:<{12}.{5}} {u_error:<{12}.{5}} {v_error:<{12}.{5}} {w_error:<{12}.{5}} {0.0:<{12}.{5}}\n")
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
    if os.path.isfile(run.c.model_out_dir+'/*.pkl'):
        print('continuing from last checkpoint')
        checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        num_ext = lambda x: int(x.split('_')[-1].split('.')[0])
        num = num_ext(checkpoint_list[-1]) + 1
        with open(checkpoint_list[-1],"rb") as f:
            model_params = pickle.load(f)
        run.train(num, model_params)
    else:
        run.train()

    
    #run.train()
