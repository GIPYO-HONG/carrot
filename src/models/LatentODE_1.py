########## module import ##########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

#BaseExperiment, make_logger, beta_generate, get_data, SEIAR, plotting
from .utiles import *

########## model define ##########

class Dynamics(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim, width_size, depth, *, key):
        self.scale = jnp.array(0.1)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim+5,
            out_size=hidden_dim,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.softplus(x),
            final_activation=lambda x: jnn.tanh(0.0001*x),
            key=key,
        )
    
    def __call__(self, t, h, args=None):
        return self.scale * self.mlp(h)
    
class LatentODE(eqx.Module):
    hidden_dyn: Dynamics
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden : eqx.nn.Linear

    hidden_to_beta: eqx.nn.Linear

    hidden_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    scales: jnp.ndarray = eqx.field(static=True)

    def __init__(self, data_size, latent_dim, hidden_dim, width_size, depth, scales, *, key):
        dyn_key, GRU_key, htl_key, lth_key, htb_key = jr.split(key, 5)

        self.hidden_dyn = Dynamics(hidden_dim, width_size, depth, key=dyn_key)
        self.rnn_cell = eqx.nn.GRUCell(data_size+1, hidden_dim, key=GRU_key)

        self.hidden_to_latent = eqx.nn.Linear(hidden_dim, latent_dim, key=htl_key)
        self.latent_to_hidden = eqx.nn.Linear(latent_dim, hidden_dim, key=lth_key)

        self.hidden_to_beta = eqx.nn.Linear(hidden_dim, 1, key=htb_key)

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.scales = scales

    def get_beta(self, h):
        beta = jnn.sigmoid(self.hidden_to_beta(h))
        return beta.squeeze()
    
    def RHS(self, t, y, args=None):

        state, h = y

        # denormalize state
        scales_state = state * self.scales

        S, E, I, A, R = scales_state

        beta = self.get_beta(h)

        kk, aa, ii, p, f, ee, dd, q = 0.526, 0.244, 0.244, 0.667, 0.98, 0.0, 1.0, 0.5

        LL = ee * E + (1 - q) * I + dd * A

        dS = -beta * S * LL
        dE = beta * S * LL - kk * E
        dI = p * kk * E - aa * I
        dA = (1 - p) * kk * E - ii * A
        dR = f * aa * I + ii * A

        scales_dstate = jnp.array([dS, dE, dI, dA, dR])

        # normalize derivative
        dstate = scales_dstate / self.scales

        dh = self.hidden_dyn(t, jnp.concatenate([h, scales_state]), args)

        return (dstate, dh)
    
    def encoder(self, ts, ys):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_dim,))
        def rnn_step(h, x):
            return self.rnn_cell(x, h), None
        hidden, _ = jax.lax.scan(rnn_step, hidden, jnp.flip(data, axis=0))
        latent = self.hidden_to_latent(hidden)
        return latent
    
    def decoder(self, latent, y0, ts):
        y0_norm = y0 / self.scales
        h0 = self.latent_to_hidden(latent)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=(y0_norm, h0),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=50000,
        )

        scales_states, h = sol.ys

        # denormalize output
        states = scales_states * self.scales

        return states, h
    
    def __call__(self, ts, ys):
        y0 = ys[0,:]

        latent = self.encoder(ts, ys)
        states, h = self.decoder(latent, y0, ts)

        return states, h

########## Experiment ##########
class Experiment(BaseExperiment):

    def __init__(self, ts, ys, beta, latent_dim=16, hidden_dim=8, width_size=64, depth=2, **kwargs):

        seed = kwargs.get("seed", 5678)

        # normalization scale
        self.scales = jnp.max(ys, axis=0) + 1e-6

        data_size=ys.shape[-1]

        model = LatentODE(
            data_size,
            latent_dim,
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

        pred, _ = model(ts, ys)

        loss = jnp.mean((pred - ys) ** 2)

        return loss
    
########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    y0, ts_data, ys_data, model = EX.y0, EX.ts, EX.ys, EX.model

    ys_eval = get_data(ts_eval, y0, EX.beta)

    latent = model.encoder(ts_data, ys_data)
    ys_pred, h_pred = model.decoder(latent, y0, ts_eval)

    beta_eval = EX.beta(ts_eval)
    beta_pred = jax.vmap(model.get_beta)(h_pred)

    plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, beta_eval, beta_pred, loss_list, viz_data)
