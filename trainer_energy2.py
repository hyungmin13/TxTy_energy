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
        update = PINN_update.lower(model_state, optimiser_fn, equation_fn, dynamic_param, dynamic_param2, static_params, static_params2, 
                                   static_keys, static_keys2, g_batch, p_batch, v_batch, Tx_batch, Ty_batch, b_batches, model_fn, model_fn2).compile()
@partial(jax.jit, static_argnums=(1, 2, 7, 8, 15, 16))
def PINN_update(model_state, optimiser_fn, equation_fn, dynamic_param, dynamic_param2, static_params, static_params2, static_keys, static_keys2, grids, particles, particle_vel, particle_Tx, particle_Ty, particle_bd, model_fn, model_fn2):
    static_leaves, treedef = static_keys
    static_leaves2, treedef2 = static_keys2
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    leaves2 = [d if s is None else s for d, s in zip(static_params2, static_leaves2)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)

    all_params2 = jax.tree_util.tree_unflatten(treedef2, leaves2)
    lossval, grads = value_and_grad(equation_fn, argnums=0)(dynamic_param, dynamic_param2, all_params, all_params2, grids, particles, particle_vel, particle_bd, particle_Tx, particle_Ty, model_fn, model_fn2)
    updates, model_state = optimiser_fn(grads, model_state, dynamic_param)
    dynamic_param = optax.apply_updates(dynamic_param, updates)
    return lossval, model_state, dynamic_param

class PINN_velbase:
    def __init__(self,c):
        c.get_outdirs()
        self.c=c

class PINNbase:
    def __init__(self,c, c2):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c
        self.c2 = c2

class PINN_velnet(PINN_velbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "problem":{}}
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

class PINN(PINNbase):
    def train(self,numb=0,**kwargs):
        all_params = {"domain":{}, "data":{}, "network1":{}, "problem":{}}
        all_params2 = {"domain":{}, "data":{}, "network1":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        all_params2["domain"] = self.c2.domain.init_params(**self.c2.domain_init_kwargs)
        all_params2["data"] = self.c2.data.init_params(**self.c2.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network1"] = self.c.network1.init_params(**self.c.network1_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        all_params2["network1"] = self.c2.network1.init_params(**self.c2.network1_init_kwargs)
        all_params2["problem"] = self.c2.problem.init_params(**self.c2.problem_init_kwargs)
        model_fn2 = self.c2.network1.network_fn
        checkpoint_list = sorted(glob(self.c2.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        with open(checkpoint_list[-1],"rb") as f:
            model_params = pickle.load(f)


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

        # Define equation function

        # Input data and grids
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        _, all_params2 = self.c2.domain.sampler(all_params2)
        _, all_params2 = self.c2.data.train_data(all_params2)
        model = Model(all_params2["network1"]["layers"], model_fn2)
        all_params2["network1"]["layers"] = from_state_dict(model, model_params).params

        dynamic_param = all_params["network1"].pop("layers")
        dynamic_param2 = all_params2["network1"].pop("layers")

        if "path_s" in all_params['problem'].keys():
            valid_data = self.c.problem.exact_solution(all_params.copy())
        else:
            valid_data = train_data.copy()
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
        #print('All_pa : ', all_params2)
        leaves2, treedef2 = jax.tree_util.tree_flatten(all_params2)
        static_params2 = tuple(x if isinstance(x,(np.ndarray, jnp.ndarray)) else None for x in leaves2)
        static_leaves2 = tuple(None if isinstance(x,(np.ndarray, jnp.ndarray)) else x for x in leaves2)
        static_keys2 = (static_leaves2, treedef2)
        # Initializing batches
        N_p = train_data['pos'].shape[0]
        perm_p = random.permutation(keys_next[0], N_p)
        data_p = []
        data_v = []
        data_Tx = []
        data_Ty = []
        b_batches = []
        print(train_data.keys())
        for i in tqdm(range(N_p//self.c.optimization_init_kwargs["p_batch"])):
            batch_p = train_data['pos'][perm_p[i*self.c.optimization_init_kwargs["p_batch"]:(i+1)*self.c.optimization_init_kwargs["p_batch"]],:]
            batch_v = train_data['vel'][perm_p[i*self.c.optimization_init_kwargs["p_batch"]:(i+1)*self.c.optimization_init_kwargs["p_batch"]],:]
            batch_Tx = train_data['Tx'][perm_p[i*self.c.optimization_init_kwargs["p_batch"]:(i+1)*self.c.optimization_init_kwargs["p_batch"]],0:1]
            batch_Ty = train_data['Ty'][perm_p[i*self.c.optimization_init_kwargs["p_batch"]:(i+1)*self.c.optimization_init_kwargs["p_batch"]],0:1]
            data_p.append(batch_p)
            data_v.append(batch_v)
            data_Tx.append(batch_Tx)
            data_Ty.append(batch_Ty)
        data_p.append(train_data['pos'][perm_p[-1-self.c.optimization_init_kwargs["p_batch"]:-1],:])
        data_v.append(train_data['vel'][perm_p[-1-self.c.optimization_init_kwargs["p_batch"]:-1],:])
        data_Tx.append(train_data['Tx'][perm_p[-1-self.c.optimization_init_kwargs["p_batch"]:-1], 0:1])
        data_Ty.append(train_data['Ty'][perm_p[-1-self.c.optimization_init_kwargs["p_batch"]:-1], 0:1])
        p_batches = itertools.cycle(data_p)
        v_batches = itertools.cycle(data_v)
        Tx_batches = itertools.cycle(data_Tx)
        Ty_batches = itertools.cycle(data_Ty)
        p_batch = next(p_batches)
        v_batch = next(v_batches)
        Tx_batch = next(Tx_batches)
        Ty_batch = next(Ty_batches)
        print(np.max(Tx_batch))
        print(np.max(Ty_batch))
        print(np.max(train_data['Tx']), np.max(train_data['Ty']))
        print(np.max(p_batch[:,0]), np.max(p_batch[:,1]), np.max(p_batch[:,2]), np.max(p_batch[:,3]))
        if "path_s" in all_params['problem'].keys():
            grids['eqns']['x'] = np.unique(valid_data['pos'][:,1:2])
            grids['eqns']['y'] = np.unique(valid_data['pos'][:,2:3])
            grids['eqns']['z'] = np.unique(valid_data['pos'][:,3:4])
            
            grids['bczu']['x'] = np.unique(valid_data['pos'][:,1:2])
            grids['bczu']['y'] = np.unique(valid_data['pos'][:,2:3])
            grids['bczl']['x'] = np.unique(valid_data['pos'][:,1:2])
            grids['bczl']['y'] = np.unique(valid_data['pos'][:,2:3])
        else:
            print('grid data and boundary data are based on linspace')
        print(np.max(grids['eqns']['t']), np.max(grids['eqns']['x']), np.max(grids['eqns']['y']), np.max(grids['eqns']['z']))
        g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                           grids['eqns'][arg], 
                                           shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                             for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)

        print(grids[all_params["domain"]["bound_keys"][0]])
        for b_key in all_params["domain"]["bound_keys"]:
            b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                            grids[b_key][arg], 
                                            shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            b_batches.append(b_batch)

        # Initializing the update function
        update = PINN_update.lower(model_state, optimiser_fn, equation_fn, dynamic_param, dynamic_param2, static_params, static_params2, 
                                   static_keys, static_keys2, g_batch, p_batch, v_batch, Tx_batch, Ty_batch, b_batches, model_fn, model_fn2).compile()
        # Training loop
        for i in range(self.c.optimization_init_kwargs["n_steps"]):
            keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
            p_batch = next(p_batches)
            v_batch = next(v_batches)
            Tx_batch = next(Tx_batches)
            Ty_batch = next(Ty_batches)
            #print(np.max(Tx_batch))
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
            #print('test')
            lossval, model_state, dynamic_param = update(model_state, dynamic_param, dynamic_param2, static_params, static_params2, g_batch, p_batch, 
                                                         v_batch, Tx_batch, Ty_batch, b_batches)
        
            #print('test') 
            self.report(numb+i, report_fn, dynamic_param, dynamic_param2, all_params, all_params2, p_batch, v_batch, g_batch, Tx_batch, Ty_batch, b_batches, valid_data, keys_iter[-1], self.c.optimization_init_kwargs["save_step"], model_fn, model_fn2)
            self.save_model(numb+i, dynamic_param, all_params, self.c.optimization_init_kwargs["save_step"], model_fn)

    def save_model(self, i, dynamic_params, all_params, save_step, model_fns):
        model_save = (i % save_step == 0)
        if model_save:
            all_params["network1"]["layers"] = dynamic_params
            model = Model(all_params["network1"]["layers"], model_fns)
            serialised_model = to_state_dict(model)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
        return

    def report(self, i, report_fn, dynamic_params, dynamic_params2, all_params, all_params2, p_batch, v_batch, g_batch, Tx_batch, Ty_batch, b_batch, valid_data, e_batch_key, save_step, model_fns, model_fns2):
        save_report = (i % save_step == 0)
        if save_report:
            all_params["network1"]["layers"] = dynamic_params
            all_params2["network1"]["layers"] = dynamic_params2
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))

            if 'T' in valid_data.keys():
                e_batch_T = random.choice(e_key, valid_data['T'], shape = (self.c.optimization_init_kwargs["e_batch"],))
                Losses = report_fn(dynamic_params, dynamic_params2, all_params, all_params2, g_batch, p_batch, v_batch, e_batch_pos, e_batch_vel, Tx_batch, Ty_batch, b_batch, model_fns, model_fns2, e_batch_T)
            else:
                print('check')
                #e_batch_T = random.choice(e_key, valid_data['T'], shape = (self.c.optimization_init_kwargs["e_batch"],))
                Losses = report_fn(dynamic_params, dynamic_params2, all_params, all_params2, g_batch, p_batch, v_batch, e_batch_pos, e_batch_vel, b_batch, model_fns, model_fns2, Tx_batch, Ty_batch)
                print('check3')
            print(f"step_num : {i:<{12}} total_loss : {Losses[0]:<{12}.{5}} u_loss : {Losses[1]:<{12}.{5}} "
                    f"v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} con_loss : {Losses[4]:<{12}.{5}} "
                    f"NS1_loss : {Losses[5]:<{12}.{5}} NS2_loss : {Losses[6]:<{12}.{5}} NS3_loss : {Losses[7]:<{12}.{5}} Eng_loss : {Losses[8]:<{12}.{5}} "
                    f"Tbu_loss : {Losses[9]:<{12}.{5}} Tbb_loss : {Losses[10]:<{12}.{5}} Tx_loss : {Losses[11]:<{12}.{5}} Ty_loss : {Losses[12]:<{12}.{5}}"
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

    with open(os.path.dirname(cur_dir)+ '/' + os.path.dirname(os.path.dirname(data['run'])) + '/' + data['network2_init_kwargs']['model_name'] +'/summary/constants.pickle','rb') as f:
        constants = pickle.load(f)
    values = list(constants.values())
    c = Constants(**data)
    c2 = Constants(run = values[0],
                        domain_init_kwargs = values[1],
                        data_init_kwargs = values[2],
                        network1_init_kwargs = values[3],
                        problem_init_kwargs = values[5],
                        optimization_init_kwargs = values[6],
                        equation_init_kwargs = values[7],)

    run = PINN(c, c2)

    """
    if os.path.isfile(run.c.model_out_dir+'saved_dic_20000.pkl'):
        print('continuing from last checkpoint')
        checkpoint_list = sorted(glob(run.c.model_out_dir+'*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        num_ext = lambda x: int(x.split('_')[-1].split('.')[0])
        num = num_ext(checkpoint_list[-1]) + 1
        with open(checkpoint_list[-1],"rb") as f:
            model_params = pickle.load(f)
        run.train(num, model_params)
    
    else:
        run.train()
    """
    
    run.train()
