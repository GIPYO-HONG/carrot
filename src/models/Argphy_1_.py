######## module import ########
import jax
import jax. random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

#BaseExperiment, make_logger, beta_generate, get_data, SEIAR, plotting
from .utiles import *

######## model define ########

class Dynamics(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim, width_size, depth, *, key):
        self.scale = jnp.array(0.1)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.softplus(x),
            final_activation=lambda x: jnn.tanh(0.0001*x),
            key=key,
        )
    
    def __call__(self, t, h, args=None):
        return self.scale * self.mlp(h)
    
class Argphy(eqx.Module):
    hidden_dyn: Dynamics
    hidden_vec: jnp.ndarray
    hidden_to_beta: eqx.nn.Linear

    scales: jnp.ndarray = eqx.field(static=True)
    
    def __init__(self, hidden_dim, width_size, depth, scales, *, key):
        dyn_key, htb_key, hvec_key = jr.split(key, 3)

        self.hidden_dyn = Dynamics(hidden_dim, width_size, depth, key=dyn_key)
        self.hidden_vec = 0.01 * jr.normal(hvec_key, (hidden_dim,))
        self.hidden_to_beta = eqx.nn.Linear(hidden_dim, 1, key=htb_key)

        self.scales = scales

    def get_beta(self, h):
        beta = jnn.sigmoid(self.hidden_to_beta(h))
        return beta.squeeze()
    
    def RHS(self, t, y, args=None):

        state, h = y

        S, E, I, A, R = state

        beta = self.get_beta(h)

        kk, aa, ii, p, f, ee, dd, q = 0.526, 0.244, 0.244, 0.667, 0.98, 0.0, 1.0, 0.5

        LL = ee * E + (1 - q) * I + dd * A

        dS = -beta * S * LL
        dE = beta * S * LL - kk * E
        dI = p * kk * E - aa * I
        dA = (1 - p) * kk * E - ii * A
        dR = f * aa * I + ii * A

        dstate = jnp.array([dS, dE, dI, dA, dR])

        dh = self.hidden_dyn(t, h, args)

        return (dstate, dh)
    
    def __call__(self, y0, ts):

        h0 = self.hidden_vec

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=(y0, h0),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=50000,
        )

        states, h = sol.ys

        return states, h

########## Experiment ##########
class Experiment(BaseExperiment):

    def __init__(self, ts, ys, beta, hidden_dim=8, width_size=64, depth=2, **kwargs):

        seed = kwargs.get("seed", 5678)

        # normalization scale
        self.scales = jnp.max(ys, axis=0) + 1e-6

        model = Argphy(
            hidden_dim,
            width_size,
            depth,
            scales=self.scales,
            key=jax.random.PRNGKey(seed)
        )

        super().__init__(model, ts, ys, **kwargs)

        self.y0 = ys[0]
        self.beta = beta

    def loss_fn(self, model, ts, ys):

        pred, _ = model(self.y0, ts)

        loss = jnp.mean(jnp.square((pred - ys) / self.scales))

        return loss
    
########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    y0, ts_data, ys_data, beta, model = EX.y0, EX.ts, EX.ys, EX.beta, EX.model

    ys_eval = get_data(ts_eval, y0, beta)
    ys_pred, h_pred = model(y0, ts_eval)

    beta_eval = EX.beta(ts_eval)
    beta_pred = jax.vmap(model.get_beta)(h_pred)

    plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, beta_eval, beta_pred, loss_list, viz_data)