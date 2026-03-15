########## module import ##########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

from .utiles import *

from datetime import datetime

########## model define ##########

# dynamics term of hidden state
class Func(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(self, scale, mlp):
        self.scale = scale
        self.mlp = mlp
    
    def __call__(self, t, z, args=None):
        tvec = jnp.array([t])
        input = jnp.concatenate([tvec, z])
        return self.scale * self.mlp(input)
    
class LatentODE(eqx.Module):
    func: Func
    rnn_cell: eqx.nn.GRUCell
    hidden_to_latent: eqx.nn.Linear
    hidden_to_beta: eqx.nn.Linear
    latent_to_initial: eqx.nn.Linear
    latent_to_hidden: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    latent_size: int = eqx.field(static=True)

    y0_use: bool = eqx.field(static=True)
    y0: jnp.ndarray = eqx.field(static=True)

    def __init__(self, *, y0, data_size, hidden_size, latent_size, width_size, depth, y0_use=False, key):
        mlp_key, GRU_key, htl_key, ltb_key, lti_key, lth_key = jr.split(key, 6)

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size+1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.softplus(x),
            final_activation=lambda x: 1e-1*jnn.tanh(0.0001*x),
            key=mlp_key,
        )
        self.func = Func(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size+1, hidden_size, key=GRU_key)
        self.hidden_to_latent = eqx.nn.Linear(hidden_size, latent_size, key=htl_key)
        self.hidden_to_beta = eqx.nn.Linear(hidden_size, 1, key=ltb_key)
        self.latent_to_initial = eqx.nn.Linear(latent_size, 5, key=lti_key)
        self.latent_to_hidden = eqx.nn.Linear(latent_size, hidden_size, key=lth_key)
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.y0_use = y0_use
        self.y0 = y0


    def encoder(self, ts, ys):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        def rnn_step(h, x):
            return self.rnn_cell(x, h), None
        hidden, _ = jax.lax.scan(rnn_step, hidden, jnp.flip(data, axis=0))
        latent = self.hidden_to_latent(hidden)
        return latent
    
    def beta(self, h):
        raw = self.hidden_to_beta(h)
        return jnp.reshape(jnn.sigmoid(0.01*raw), ())
    
    def initial(self, latent):
        raw = self.latent_to_initial(latent)
        return jnn.softmax(raw)

    def decoder(self, ts, ys):
        z0 = self.encoder(ts, ys)
        h0 = self.latent_to_hidden(z0)
        if self.y0_use:
            y0 = self.y0
        else:
            y0 = self.initial(z0)
        
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.dynamics),
            diffrax.Tsit5(),
            t0=0,
            t1=ts[-1],
            dt0=0.1,
            y0=(y0, h0),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            max_steps=50000,
        )
        seiar_traj, h_traj = sol.ys
        return seiar_traj, h_traj, z0

    def dynamics(self, t, y, args=None):
        seiar, h = y
        S, E, I, A, R = seiar
        beta_t = self.beta(h)
        kk, aa, ii, p, f, ee, dd, q = 0.526, 0.244, 0.244, 0.667, 0.98, 0.0, 1.0, 0.5
        LL = ee * E + (1 - q) * I + dd * A
        
        dS = -beta_t * S * LL
        dE = beta_t * S * LL - kk * E
        dI = p * kk * E - aa * I
        dA = (1 - p) * kk * E - ii * A
        dR = f * aa * I + ii * A
        
        dh = self.func(t, h, args)
        return (jnp.array([dS, dE, dI, dA, dR]), dh)

    def evaluation(self, ts, latent):
        h0 = self.latent_to_hidden(latent)

        if self.y0_use:
            y0 = self.y0
        else:
            y0 = self.initial(latent)
            
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.dynamics),
            diffrax.Tsit5(),
            t0=0, t1=ts[-1], dt0=0.1,
            y0=(y0, h0), saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

########## Experiment ##########
class Experiment(BaseExperiment):
    def __init__(self, beta, y0, ts, ys, I_only=False, latent_dim=8, hidden_dim=4, width_size=64, depth=2, y0_use=False, **kwargs):
        seed = kwargs.get('seed', 5678)
        model = LatentODE(
            y0=y0,
            data_size=ys.shape[-1], 
            hidden_size=hidden_dim, 
            latent_size=latent_dim, 
            width_size=width_size, 
            depth=depth, 
            y0_use=y0_use,
            key=jr.PRNGKey(seed)
        )
        
        super().__init__(model, y0, ts, ys, **kwargs)
        
        self.I_only = I_only
        self.beta = beta
        
        if self.I_only:
            self.scales = jnp.max(ys) + 1e-4
        else:
            self.scales = jnp.max(ys, axis=0) + 1e-4

    def loss_fn(self, model, ts, ys):
        pred, _, z0 = model.decoder(ts, ys)
        
        if self.I_only:
            loss = jnp.mean(jnp.square((pred[:, 2:3] - ys)/self.scales))
        else:
            loss = jnp.mean(jnp.square((pred - ys) / self.scales))
            
        return loss, z0

########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    y0, ts_data, ys_data, model, I_only = EX.y0, EX.ts_data, EX.ys_data, EX.model, EX.I_only

    ys_eval = get_data(ts_eval, y0, EX.beta)
    latent = model.encoder(ts_data, ys_data)

    ys_pred, h_traj = model.evaluation(ts_eval, latent)

    beta_eval = EX.beta(ts_eval)
    beta_pred = jax.vmap(model.beta)(h_traj)

    plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, beta_eval, beta_pred, loss_list, I_only, viz_data)

########## test ##########

if __name__=='__main__':
    exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    beta = beta_generate(5e-1, 0.1, 0.)
    y0 = jnp.array([1e+0, 0., 1e-6, 0., 0.])
    ts = jnp.linspace(0., 365., 366)
    ys = get_data(ts, y0, beta.func)

    EX = Experiment(
        beta=beta.func, 
        y0=y0, 
        ts=ts, 
        ys=ys, 
        I_only=False, 
        exp_name=exp_name
    )

    EX.train(lr=1e-3, steps=10000)