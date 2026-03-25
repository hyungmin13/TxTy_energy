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
import itertools
import numpy as np
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

@partial(jax.jit, static_argnums=(1, 2, 5, 10))
def PINN_update(model_state, optimiser_fn, equation_fn, dynamic_param, static_params, static_keys, grids, particles, particle_vel, particle_bd, model_fn):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(equation_fn, argnums=0)(dynamic_param, all_params, grids, particles, particle_vel, particle_bd, model_fn)
    updates, model_state = optimiser_fn(grads, model_state, dynamic_param)
    dynamic_param = optax.apply_updates(dynamic_param, updates)
    return lossval, model_state, dynamic_param

@partial(jax.jit, static_argnums=(2, 3, 7, 12, 13))
def PINN_update2(model_states, model_states2, optimiser_fn, equation_fn, dynamic_params, dynamic_params2, static_params, static_keys, grids, particles, particle_vel, particle_bd, model_fn, model_fn2):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(equation_fn, argnums=(0,1))(dynamic_params, dynamic_params2, all_params, grids, particles, particle_vel, particle_bd, model_fn, model_fn2)
    updates, model_states = optimiser_fn(grads[0], model_states, dynamic_params)
    updates2, model_states2 = optimiser_fn(grads[1], model_states2, dynamic_params2)
    dynamic_params = optax.apply_updates(dynamic_params, updates)
    dynamic_params2 = optax.apply_updates(dynamic_params2, updates2)
    return lossval, model_states, model_states2, dynamic_params, dynamic_params2

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 9, 14, 15))
def PINN_update3(model_states, model_states2, optimiser_fn, equation_fn, equation_fn2, equation_fn3, dynamic_params, dynamic_params2, static_params, static_keys, grids, particles, particle_vel, particle_bd, model_fn, model_fn2):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(equation_fn, argnums=0)(dynamic_params, dynamic_params2, all_params, grids, particles, particle_vel, particle_bd, model_fn, model_fn2)
    Tx,Ty = equation_fn3(dynamic_params, all_params, grids, model_fn)
    lossval2, grads2 = value_and_grad(equation_fn2, argnums=1)(dynamic_params, dynamic_params2, all_params, grids, particle_bd, Tx, Ty, model_fn, model_fn2)
    updates, model_states = optimiser_fn(grads, model_states, dynamic_params)
    updates2, model_states2 = optimiser_fn(grads2, model_states2, dynamic_params2)
    dynamic_params = optax.apply_updates(dynamic_params, updates)
    dynamic_params2 = optax.apply_updates(dynamic_params2, updates2)
    return lossval, lossval2, model_states, model_states2, dynamic_params, dynamic_params2

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c

class PINN(PINNbase):
    def train(self,numb=0,**kwargs):
        all_params = {"domain":{}, "data":{}, "network1":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network1"] = self.c.network1.init_params(**self.c.network1_init_kwargs)
        try:
            all_params["network2"] = self.c.network2.init_params(**self.c.network2_init_kwargs)
        except:
            print("2nd network is not intialized")
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)

        #if 'model_params' in kwargs.keys():
        #    model_params = kwargs['model_params']
        # Initialize optmiser
        learn_rate = optax.exponential_decay(self.c.optimization_init_kwargs["learning_rate"],
                                             self.c.optimization_init_kwargs["decay_step"],
                                             self.c.optimization_init_kwargs["decay_rate"],)
        optimiser = self.c.optimization_init_kwargs["optimiser"](learning_rate=learn_rate, b1=0.95, b2=0.95,
                                                                 weight_decay=0.01, precondition_frequency=5)
        model_state = optimiser.init(all_params["network1"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = self.c.network1.network_fn
        equation_fn = self.c.equation1.Loss
        report_fn = self.c.equation1.Loss_report
        #print('check1')
        if "network2" in all_params.keys():
            model_state2 = optimiser.init(all_params["network2"]["layers"])
            optimiser_fn2 = optimiser.update
            model_fn2 = self.c.network2.network_fn2

        # Define equation function

        # Input data and grids
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        #if 'model_params' in kwargs.keys():
        #    model = Model(all_params['network']['layers'], model_fn)
        #    all_params["network"]["layers"] = from_state_dict(model, model_params).params
        dynamic_param = all_params["network1"].pop("layers")
        
        if "network2" in all_params.keys():
            dynamic_param2 = all_params["network2"].pop("layers")
        valid_data = self.c.problem.exact_solution(all_params.copy())
        if "equation2" in self.c.equation_init_kwargs.keys():
            equation_fn2 = self.c.equation2.Loss
            equation_fn3 = self.c.equation1.TxTy_cal
            report_fn2 = self.c.equation2.Loss_report

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
        N_p = train_data['pos'].shape[0]
        perm_p = random.permutation(keys_next[0], N_p)
        data_p = []
        data_v = []
        for i in range(N_p//self.c.optimization_init_kwargs["p_batch"]):
            batch_p = train_data['pos'][perm_p[i*self.c.optimization_init_kwargs["p_batch"]:(i+1)*self.c.optimization_init_kwargs["p_batch"]],:]
            batch_v = train_data['vel'][perm_p[i*self.c.optimization_init_kwargs["p_batch"]:(i+1)*self.c.optimization_init_kwargs["p_batch"]],:]
            data_p.append(batch_p)
            data_v.append(batch_v)
        data_p.append(train_data['pos'][perm_p[-1-self.c.optimization_init_kwargs["p_batch"]:-1],:])
        data_v.append(train_data['vel'][perm_p[-1-self.c.optimization_init_kwargs["p_batch"]:-1],:])
        p_batches = itertools.cycle(data_p)
        v_batches = itertools.cycle(data_v)
        p_batch = next(p_batches)
        v_batch = next(v_batches)

        grids['eqns']['x'] = np.unique(valid_data['pos'][::2,1:2])
        grids['eqns']['y'] = np.unique(valid_data['pos'][::2,2:3])
        grids['eqns']['z'] = np.unique(valid_data['pos'][:,3:4])
        try:
            print('T_ref : ', all_params["data"]['T_ref'])
        except:
            print('no T_ref')
        g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                           grids['eqns'][arg], 
                                           shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                             for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
        b_batches = []
        grids['bczu']['x'] = np.unique(valid_data['pos'][:,1:2])
        grids['bczu']['y'] = np.unique(valid_data['pos'][:,2:3])
        #grids['bczu']['z'] = np.unique(valid_data['pos'][:,3:4])
        grids['bczl']['x'] = np.unique(valid_data['pos'][:,1:2])
        grids['bczl']['y'] = np.unique(valid_data['pos'][:,2:3])
        #grids['bczl']['z'] = np.unique(valid_data['pos'][:,3:4])
        for b_key in all_params["domain"]["bound_keys"]:
            b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                            grids[b_key][arg], 
                                            shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            b_batches.append(b_batch)
        print(np.max(grids['eqns']['t']), np.max(grids['eqns']['x']), np.max(grids['eqns']['y']),np.max(grids['eqns']['z']))
        print(np.max(p_batch[:,0]), np.max(p_batch[:,1]), np.max(p_batch[:,2]), np.max(p_batch[:,3]))
        # Initializing the update function
        if "network2" in all_params.keys():
            if "equation2" in self.c.equation_init_kwargs.keys():
                update = PINN_update3.lower(model_state, model_state2, optimiser_fn, equation_fn, equation_fn2, equation_fn3, dynamic_param, 
                                            dynamic_param2, static_params, static_keys, g_batch, p_batch, v_batch, b_batches, model_fn, model_fn2).compile()
            else:
                update = PINN_update2.lower(model_state, model_state2, optimiser_fn, equation_fn, dynamic_param, 
                                            dynamic_param2, static_params, static_keys, g_batch, p_batch, v_batch, b_batches, model_fn, model_fn2).compile()
        else:
            update = PINN_update.lower(model_state, optimiser_fn, equation_fn, dynamic_param, static_params, static_keys, 
                                       g_batch, p_batch, v_batch, b_batches, model_fn).compile()
        
        # Training loop
        if "network2" in all_params.keys():
            if "equation2" in self.c.equation_init_kwargs.keys():
                for i in range(self.c.optimization_init_kwargs["n_steps"]):
                    keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
                    p_batch = next(p_batches)
                    v_batch = next(v_batches)
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
                    lossval, lossval2, model_state, model_state2, dynamic_param, dynamic_param2 = update(model_state, model_state2, dynamic_param, dynamic_param2, static_params, 
                                                                                                g_batch, p_batch, v_batch, b_batches)
                
                
                    self.report2(numb+i, report_fn, dynamic_param, dynamic_param2, all_params, p_batch, 
                                    v_batch, g_batch, b_batch, valid_data, keys_iter[-1], self.c.optimization_init_kwargs["save_step"], model_fn, model_fn2)
                    self.save_model2(numb+i, dynamic_param, dynamic_param2, all_params, self.c.optimization_init_kwargs["save_step"], model_fn, model_fn2)
            else:
                for i in range(self.c.optimization_init_kwargs["n_steps"]):
                    keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
                    p_batch = next(p_batches)
                    v_batch = next(v_batches)
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
                    lossval, model_state, model_state2, dynamic_param, dynamic_param2 = update(model_state, model_state2, dynamic_param, dynamic_param2, static_params, 
                                                                                                g_batch, p_batch, v_batch, b_batches)
                
                
                    self.report2(numb+i, report_fn, dynamic_param, dynamic_param2, all_params, p_batch, 
                                    v_batch, g_batch, b_batch, valid_data, keys_iter[-1], self.c.optimization_init_kwargs["save_step"], model_fn, model_fn2)
                    self.save_model2(numb+i, dynamic_param, dynamic_param2, all_params, self.c.optimization_init_kwargs["save_step"], model_fn, model_fn2)
        else:
            print('lets go')
            for i in range(self.c.optimization_init_kwargs["n_steps"]):
                keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
                p_batch = next(p_batches)
                v_batch = next(v_batches)
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
                lossval, model_state, dynamic_param = update(model_state, dynamic_param, static_params, g_batch, p_batch, v_batch, b_batches)
            
            
                self.report(numb+i, report_fn, dynamic_param, all_params, p_batch, v_batch, g_batch, b_batch, valid_data, keys_iter[-1], self.c.optimization_init_kwargs["save_step"], model_fn)
                self.save_model1(numb+i, dynamic_param, all_params, self.c.optimization_init_kwargs["save_step"], model_fn)

    def save_model1(self, i, dynamic_params, all_params, save_step, model_fns):
        model_save = (i % save_step == 0)
        if model_save:
            all_params["network1"]["layers"] = dynamic_params
            model = Model(all_params["network1"]["layers"], model_fns)
            serialised_model = to_state_dict(model)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
        return

    def save_model2(self, i, dynamic_params, dynamic_params2, all_params, save_step, model_fns, model_fns2):
        model_save = (i % save_step == 0)
        if model_save:
            all_params["network1"]["layers"] = dynamic_params
            all_params["network2"]["layers"] = dynamic_params2
            model = Model(all_params["network1"]["layers"], model_fns)
            model2 = Model(all_params["network2"]["layers"], model_fns2)
            serialised_model = to_state_dict(model)
            serialised_model2 = to_state_dict(model2)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
            with open(self.c.model_out_dir2 + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model2,f)
        return

    def report(self, i, report_fn, dynamic_params, all_params, p_batch, v_batch, g_batch, b_batch, valid_data, e_batch_key, save_step, model_fns):
        save_report = (i % save_step == 0)
        if save_report:
            all_params["network1"]["layers"] = dynamic_params
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            print(np.max(e_batch_pos[:,0]), np.max(e_batch_pos[:,1]), np.max(e_batch_pos[:,2]), np.max(e_batch_pos[:,3]))
            if 'T' in valid_data.keys():
                e_batch_T = random.choice(e_key, valid_data['T'], shape = (self.c.optimization_init_kwargs["e_batch"],))
                Losses = report_fn(dynamic_params, all_params, g_batch, p_batch, v_batch, e_batch_pos, e_batch_vel, b_batch, model_fns, e_batch_T)
            else:
                Losses = report_fn(dynamic_params, all_params, g_batch, p_batch, v_batch, e_batch_pos, e_batch_vel, b_batch, model_fns)


            print(f"step_num : {i:<{12}} total_loss : {Losses[0]:<{12}.{5}} u_loss : {Losses[1]:<{12}.{5}} "
                      f"v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} con_loss : {Losses[4]:<{12}.{5}} "
                      f"NS1_loss : {Losses[5]:<{12}.{5}} NS2_loss : {Losses[6]:<{12}.{5}} NS3_loss : {Losses[7]:<{12}.{5}} Eng_loss : {Losses[8]:<{12}.{5}} "
                      f"Tbu_loss : {Losses[9]:<{12}.{5}} Tbb_loss : {Losses[10]:<{12}.{5}} "
                      f"u_error : {Losses[11]:<{12}.{5}} v_error : {Losses[12]:<{12}.{5}} w_error : {Losses[13]:<{12}.{5}} T_error : {Losses[14]:<{12}.{5}}")
            with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} "
                            f"{Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {Losses[7]:<{12}.{5}} {Losses[8]:<{12}.{5}} {Losses[11]:<{12}.{5}} {Losses[12]:<{12}.{5}} {Losses[13]:<{12}.{5}} {Losses[14]:<{12}.{5}}\n")
            f.close()
        return
    def report2(self, i, report_fn, dynamic_params, dynamic_params2, all_params, p_batch, v_batch, g_batch, b_batch, valid_data, e_batch_key, save_step, model_fns, model_fns2):
        save_report = (i % save_step == 0)
        if save_report:
            all_params["network1"]["layers"] = dynamic_params
            all_params["network2"]["layers"] = dynamic_params2
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            if 'T' in valid_data.keys():
                e_batch_T = random.choice(e_key, valid_data['T'], shape = (self.c.optimization_init_kwargs["e_batch"],))
                Losses = report_fn(dynamic_params, dynamic_params2, all_params, g_batch, p_batch, v_batch, e_batch_pos, e_batch_vel, e_batch_T, b_batch, model_fns, model_fns2)
            else:
                Losses = report_fn(dynamic_params, dynamic_params2, all_params, g_batch, p_batch, v_batch, e_batch_pos, e_batch_vel, b_batch, model_fns, model_fns2)
            if 'T' in valid_data.keys():
                print(f"step_num : {i:<{12}} total_loss : {Losses[0]:<{12}.{5}} u_loss : {Losses[1]:<{12}.{5}} "
                      f"v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} con_loss : {Losses[4]:<{12}.{5}} "
                      f"NS1_loss : {Losses[5]:<{12}.{5}} NS2_loss : {Losses[6]:<{12}.{5}} NS3_loss : {Losses[7]:<{12}.{5}} Eng_loss : {Losses[8]:<{12}.{5}} "
                      f"Tbu_loss : {Losses[9]:<{12}.{5}} Tbb_loss : {Losses[10]:<{12}.{5}} "
                      f"u_error : {Losses[11]:<{12}.{5}} v_error : {Losses[12]:<{12}.{5}} w_error : {Losses[13]:<{12}.{5}} T_error : {Losses[14]:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                     f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} "
                            f"{Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {Losses[7]:<{12}.{5}} {Losses[8]:<{12}.{5}} {Losses[11]:<{12}.{5}} {Losses[12]:<{12}.{5}} {Losses[13]:<{12}.{5}} {Losses[14]:<{12}.{5}}\n")
            else:
                print(f"step_num : {i:<{12}} total_loss : {Losses[0]:<{12}.{5}} u_loss : {Losses[1]:<{12}.{5}} "
                      f"v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} con_loss : {Losses[4]:<{12}.{5}} "
                      f"NS1_loss : {Losses[5]:<{12}.{5}} NS2_loss : {Losses[6]:<{12}.{5}} NS3_loss : {Losses[7]:<{12}.{5}} "
                      f"u_error : {Losses[8]:<{12}.{5}} v_error : {Losses[9]:<{12}.{5}} w_error : {Losses[10]:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} "
                            f"{Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {0.0:<{12}.{5}} {0.0:<{12}.{5}} {0.0:<{12}.{5}} {Losses[8]:<{12}.{5}} {Losses[9]:<{12}.{5}} {Losses[10]:<{12}.{5}} {0.0:<{12}.{5}}\n")
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
    print('check1')
    input_txt = cur_dir + '/' + args.config + '.txt' 
    data = parse_tree_structured_txt(input_txt)
    c = Constants(**data)
    print('check2')
    run = PINN(c)
    if os.path.isfile(run.c.model_out_dir+'saved_dic_20000.pkl'):
        print('continuing from last checkpoint')
        checkpoint_list = sorted(glob(run.c.model_out_dir+'*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        num_ext = lambda x: int(x.split('_')[-1].split('.')[0])
        num = num_ext(checkpoint_list[-1]) + 1
        with open(checkpoint_list[-1],"rb") as f:
            model_params = pickle.load(f)
        run.train(num, model_params)
    else:
        print('check3')
        run.train()

    
    #run.train()
