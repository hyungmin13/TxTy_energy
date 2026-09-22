#%%
import jax.numpy as jnp
from jax import random
import numpy as np
class Network:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError

class MLP(Network):
    def __init__(self, all_params):
        self.all_params = all_params
        

    @staticmethod
    def init_params(key, layer_sizes, network_name):
        key_network = random.PRNGKey(key)
        keys = random.split(key_network, len(layer_sizes)-1)
        params = [eval(network_name)._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        network_params = {"layers": params, "layer_sizes":layer_sizes, "network_name":network_name}
        return network_params
    
    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        w = random.normal(w_key, (m, n))
        b = jnp.zeros((1,n))
        g = jnp.ones((1,n))
        return w,b,g

    @staticmethod
    def network_fn(all_params, x):
        params = all_params["network1"]["layers"]
        inmin = all_params["domain"]["in_min"]
        inmax = all_params["domain"]["in_max"]
        inmean = all_params["data"]["in_mean"]
        instd = all_params["data"]["in_std"]
        x = 2*(x - inmin)/(inmax-inmin) - 1
        #x = (x-inmean)/instd
        for w, b, g in params[:-1]:
            x = g*jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
            x = jnp.tanh(x)
        w, b, g = params[-1]
        x = jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
        return x

    def network_fn3(self, x):
        params = self.all_params["network2"]["layers"]
        inmin = self.all_params["domain"]["in_min"]
        inmax = self.all_params["domain"]["in_max"]
        x = 2*(x - inmin)/(inmax-inmin) - 1
        for w, b, g in params[:-1]:
            x = g*jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
            x = jnp.tanh(x)
        w, b, g = params[-1]
        x = jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
        return x

class MLP_FF(Network):
    def __init__(self, all_params):
        self.all_params = all_params

    @staticmethod
    def init_params(key, layer_sizes, network_name, input_dim=4, ff_dim_t=16, ff_dim_s=64, ff_sigma_t=0.01, ff_sigma_s=0.01):
        key_network = random.PRNGKey(key)
        key_B, key_layers = random.split(key_network)
        key_B_t, key_B_s = random.split(key_B)
        # Fourier projection matrix
        # shape: (input_dim, ff_dim)
        B_t = ff_sigma_t * random.normal(key_B_t, (1, ff_dim_t))
        B_s = ff_sigma_s * random.normal(key_B_s, (3, ff_dim_s))
        keys = random.split(key_layers, len(layer_sizes) - 1)
        params = [eval(network_name)._random_layer_params(keys[0],ff_dim_s*2+ff_dim_t*2+layer_sizes[0],layer_sizes[1])]+ [eval(network_name)._random_layer_params(k, m, n) for k, m, n in zip(keys[1:], layer_sizes[1:-1], layer_sizes[2:])]

        network_params = {
            "layers": params,
            "layer_sizes": layer_sizes,
            "network_name": network_name,
            "B_t": B_t,
            "B_s": B_s,
            "ff_dim_t": ff_dim_t,
            "ff_dim_s": ff_dim_s,
            "ff_sigma_t": ff_sigma_t,
            "ff_sigma_s": ff_sigma_s,
        }
        return network_params

    @staticmethod
    def _random_layer_params(key, m, n):
        w_key, b_key = random.split(key)
        w = random.normal(w_key, (m, n))
        b = jnp.zeros((1, n))
        g = jnp.ones((1, n))
        return w, b, g

    @staticmethod
    def fourier_features(x, B_t, B_s):
        # x shape: (N, input_dim)
        # B shape: (input_dim, ff_dim)
        t = x[:,0:1]
        s = x[:,1:4]
        #x_proj = 2*jnp.pi*(x@B)   # (N, ff_dim)
        t_proj = 2*jnp.pi*(t@B_t)
        s_proj = 2*jnp.pi*(s@B_s)
        x_ff = jnp.concatenate([x, jnp.sin(t_proj), jnp.cos(t_proj), jnp.sin(s_proj), jnp.cos(s_proj)], axis=-1)
        return x_ff

    @staticmethod
    def network_fn(all_params, x):
        net_params = all_params["network1"]
        params = net_params["layers"]
        B_t = net_params["B_t"]
        B_s = net_params["B_s"]
        inmin = all_params["domain"]["in_min"]
        inmax = all_params["domain"]["in_max"]

        # normalize to [-1, 1]
        x = 2 * (x - inmin) / (inmax - inmin) - 1

        # Fourier feature mapping
        x = MLP_FF.fourier_features(x, B_t, B_s)

        for w, b, g in params[:-1]:
            x = g * jnp.dot(x, w / jnp.linalg.norm(w, axis=0, keepdims=True)) + b
            x = jnp.tanh(x)

        w, b, g = params[-1]
        x = jnp.dot(x, w / jnp.linalg.norm(w, axis=0, keepdims=True)) + b

        return x

class MLPSiren(Network):
    def __init__(self, all_params):
        self.all_params = all_params

    @staticmethod
    def init_params(key, layer_sizes, network_name, omega0=30.0):
        key_network = random.PRNGKey(key)
        keys = random.split(key_network, len(layer_sizes) - 1)

        params = []
        for i, (k, m, n) in enumerate(zip(keys, layer_sizes[:-1], layer_sizes[1:])):
            first_layer = (i == 0)
            params.append(
                eval(network_name)._random_layer_params(k, m, n, first_layer, omega0)
            )

        network_params = {
            "layers": params,
            "layer_sizes": layer_sizes,
            "network_name": network_name,
            "omega0": omega0,
        }
        return network_params

    @staticmethod
    def _random_layer_params(key, m, n, first_layer=False, omega0=30.0):
        w_key, b_key = random.split(key)

        if first_layer:
            # SIREN first layer init
            w = random.uniform(w_key, (m, n), minval=-1.0/m, maxval=1.0/m)
        else:
            # SIREN hidden layer init
            limit = jnp.sqrt(6.0 / m) / omega0
            w = random.uniform(w_key, (m, n), minval=-limit, maxval=limit)

        b = jnp.zeros((1, n))
        return w, b

    @staticmethod
    def network_fn(all_params, x):
        net_params = all_params["network1"]
        params = net_params["layers"]
        omega0 = net_params["omega0"]

        inmin = all_params["domain"]["in_min"]
        inmax = all_params["domain"]["in_max"]

        # normalize input
        x = 2 * (x - inmin) / (inmax - inmin) - 1

        # hidden layers
        for i, (w, b) in enumerate(params[:-1]):
            if i == 0:
                x = jnp.sin(omega0 * (jnp.dot(x, w) + b))
            else:
                x = jnp.sin(jnp.dot(x, w) + b)

        # output layer: usually linear
        w, b = params[-1]
        x = jnp.dot(x, w) + b

        return x

if __name__=="__main__":
    from domain import *

    all_params = {"network":{}, "domain":{}}
    domain_range = {'t':(0,8), 'x':(0,1.2), 'y':(0,1.2), 'z':(0,1)}
    frequency = 3000
    grid_size = [9, 200, 200, 200]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']
    all_params["domain"] = Domain.init_params(domain_range, frequency, grid_size, bound_keys)
    grids, all_params = Domain.sampler(all_params)
    x = jnp.ones((100,4))
    key = random.PRNGKey(0)
    layer_sizes = [4,16,32,16,1]
    network = MLP
    all_params["network"] = network.init_params(key, layer_sizes)

