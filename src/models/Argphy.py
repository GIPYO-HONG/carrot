########## module import ##########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

from .utiles import *

########## model define ##########

# dynamics term of hidden state
class Dyn(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim, width_size, depth, *, key):
        self.scale = jnp.array(0.1)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.sigmoid(x),
            final_activation=lambda x: jnn.tanh(x), 
            key=key,
        )
    
    def __call__(self, t, h, args=None):
        # tvec = jnp.array([t])
        # input = jnp.concatenate([tvec, z])
        # return self.scale * self.mlp(input)
        return self.scale * self.mlp(h)
    
class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, final_activation=None, *, key):
        if final_activation is None:
            final_activation = lambda x: x

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=64,
            depth=2,
            activation=jnn.sigmoid,
            final_activation=final_activation, 
            key=key,
        )
    
    def __call__(self, y):
        return self.mlp(y)
    
class Argphy(eqx.Module):
    dbb: Dyn
    
    latent_to_state: Func
    latent_to_hidden: Func

    hidden_to_beta: Func

    latent_vec: jnp.ndarray

    def __init__(self, latent_dim, hidden_dim, width_size, depth, *, key):
        mlp_key, lts_key, lth_key, htb_key, vec_key= jr.split(key, 5)

        self.dbb = Dyn(hidden_dim, width_size, depth, key=mlp_key)                                # define beta dynamics

        self.latent_to_state = Func(latent_dim, 5, key=lts_key)                                   # latnet vector -> state initial
        self.latent_to_hidden = Func(latent_dim, hidden_dim, key=lth_key)                         # latent vector -> hidden initial

        self.hidden_to_beta = Func(hidden_dim, 1,
                                   final_activation=lambda x: 3.0 * jnn.sigmoid(x),
                                   key=htb_key)                                                   # hidden -> beta

        self.latent_vec = jnp.array(jr.uniform(vec_key, (latent_dim,), minval=0., maxval=1.))     # define latent vector for random
        # self.latent_vec = jnp.array([1., 0., 1e-6, 0, 0])

    # Use total population is 1 for initial value
    def initial(self, latent):
        raw = self.latent_to_state(latent)
        return jnn.softmax(raw)

    def dynamics(self, t, y, args=None):
        state, h = y
        S, E, I, A, R = state
        beta = self.hidden_to_beta(h).squeeze()

        kk, aa, ii, p, f, ee, dd, q = 0.526, 0.244, 0.244, 0.667, 0.98, 0.0, 1.0, 0.5
        LL = ee * E + (1 - q) * I + dd * A

        dS = -beta * S * LL
        dE = beta * S * LL - kk * E
        dI = p * kk * E - aa * I
        dA = (1 - p) * kk * E - ii * A
        dR = f * aa * I + ii * A
        
        dh = self.dbb(t, h, args)
        return (jnp.array([dS, dE, dI, dA, dR]), dh)
    
    def __call__(self, ts, y0_=None, y0_use=False):
        if y0_use:
            y0 = y0_
        else:
            y0 = self.initial(self.latent_vec)

        h0 = self.latent_to_hidden(self.latent_vec)
        
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.dynamics),
            diffrax.Tsit5(),
            t0=0,
            t1=ts[-1],
            dt0=0.1,
            y0=(y0, h0),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-2, atol=1e-4),
            max_steps=50000, 
        )
        return sol.ys # (seiar_traj, z_traj)
    
########## Experiment ##########
class Experiment(BaseExperiment):
    def __init__(self, y0, ts, ys, beta, latent_dim=8, hidden_dim=2, width_size=64, depth=2, I_only=False, y0_use=False, **kwargs):
        seed = kwargs.get('seed', 5678)
        model = Argphy(latent_dim, hidden_dim, width_size, depth, key=jr.PRNGKey(seed))
        
        super().__init__(model, y0, ts, ys, **kwargs)
        
        # Not use training, but save info for evaluation
        self.y0 = y0
        self.beta = beta
        
        # for variation experiments
        self.I_only = I_only
        self.y0_use = y0_use

    def loss_fn(self, model, ts, ys):
        pred, _ = model(ts, y0_=self.y0, y0_use=self.y0_use)

        eps = 1e-6

        if self.I_only:
            pred = jnp.clip(pred[:, 2:3], eps, None)
            ys = jnp.clip(ys, eps, None)

            loss = jnp.mean((jnp.log(pred) - jnp.log(ys))**2)

        else:
            pred = jnp.clip(pred, eps, None)
            ys = jnp.clip(ys, eps, None)

            loss = jnp.mean((jnp.log(pred) - jnp.log(ys))**2)

        return loss, None

########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    y0, ts_data, ys_data, beta, model, I_only = EX.y0, EX.ts, EX.ys, EX.beta, EX.model, EX.I_only

    ys_eval = get_data(ts_eval, y0, beta)
    ys_pred, z_pred = model(ts_eval)

    beta_eval = EX.beta(ts_eval)
    beta_pred = jax.vmap(model.hidden_to_beta)(z_pred)

    plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, beta_eval, beta_pred, loss_list, I_only, viz_data)