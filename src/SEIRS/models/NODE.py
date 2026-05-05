########## module import ##########
import jax
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

#BaseExperiment, make_logger, beta_generate, get_data, SEIAR, plotting
from .utiles import *


########## model define ##########

class Beta(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.softplus(x),
            final_activation=lambda x: jnn.sigmoid(0.0001*x),
            key=key
            )
        
    def __call__(self, t):
        t_input = jnp.atleast_1d(t) 
        beta_out = self.mlp(t_input)
        return beta_out.squeeze()
    
class NODE(eqx.Module):
    beta: Beta

    def __init__(self, width_size, depth, *, key):
        self.beta = Beta(width_size, depth, key=key)

    def RHS(self, t, y, args=None):

        S, E, I, A, R = y

        beta = self.beta(t)

        kk, aa, ii, p, f, ee, dd, q = 0.526, 0.244, 0.244, 0.667, 0.98, 0.0, 1.0, 0.5

        LL = ee * E + (1 - q) * I + dd * A

        dS = -beta * S * LL
        dE = beta * S * LL - kk * E
        dI = p * kk * E - aa * I
        dA = (1 - p) * kk * E - ii * A
        dR = f * aa * I + ii * A

        dstate = jnp.array([dS, dE, dI, dA, dR])

        return dstate
    
    def __call__(self, y0, ts):

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=50000,
        )

        return sol.ys
    
########## Experiment ##########
    
class Experiment(BaseExperiment):

    def __init__(self, ts, y0, ys, beta, width_size=64, depth=2, **kwargs):

        seed = kwargs.get("seed", 5678)

        # normalization scale
        self.scales = jnp.max(ys, axis=0) + 1e-6

        model = NODE(
            width_size,
            depth,
            key=jax.random.PRNGKey(seed)
        )

        super().__init__(model, ts, ys, **kwargs)

        self.y0 = y0
        self.beta = beta

    def loss_fn(self, model, ts, ys):

        pred = model(self.y0, ts)

        loss = jnp.mean(jnp.square((pred - ys) / self.scales))

        return loss
    
########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    y0, ts_data, ys_data, model = EX.y0, EX.ts, EX.ys, EX.model
    ys_eval = get_data(ts_eval, y0, EX.beta)
    ys_pred = model(y0, ts_eval)
    beta_eval = EX.beta(ts_eval)
    beta_pred = jax.vmap(lambda t: model.beta(jnp.array([t])))(ts_eval)

    plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, beta_eval, beta_pred, loss_list, viz_data)