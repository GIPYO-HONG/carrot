######## module import ########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

from .utiles import *

######## model define ########

class Dynamics(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim, width_size, depth, *, key):
        self.scale = jnp.array(0.1)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim + 17,
            out_size=hidden_dim,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.softplus(x),
            final_activation=lambda x: jnn.tanh(0.01*x),
            key=key,
        )

    def __call__(self, t, h, args=None):
        return self.scale * self.mlp(h)


class Argphy(eqx.Module):
    hidden_dyn: Dynamics
    hidden_vec: jnp.ndarray
    hidden_to_beta: eqx.nn.Linear

    y0: jnp.ndarray
    s: jnp.ndarray

    def __init__(self, hidden_dim, width_size, depth, *, key):
        dyn_key, htb_key, hvec_key= jr.split(key, 3)

        self.hidden_dyn = Dynamics(hidden_dim, width_size, depth, key=dyn_key)
        self.hidden_vec = 0.01 * jr.normal(hvec_key, (hidden_dim,))
        self.hidden_to_beta = eqx.nn.Linear(hidden_dim, 1, key=htb_key)
        
        self.y0 = jnp.log(jnp.array([0.0008417, 0.01495894, 0.00082556, 0.00068982, 0.00063456, 0.00832867,
                             0.00190631, 0.00155624, 0.00142264, 0.01858621, 0.02000239, 0.01796071,
                             0.01670121, 0.22218046, 0.02531468, 0.03337227, 0.6147068 ]))
        
        self.s = jnp.array(1000.)

    def get_beta(self, h):
        beta = 8*jnn.sigmoid(self.hidden_to_beta(h)) + 25
        return beta.squeeze()
    
    def RHS(self, t, y, args=None):
        state, h = y

        xi, mu, sigma, nu, gamma = 13 / 12, 0.041 / 12, 91 / 12, 36 / 12, 1.8 / 12

        M, S1, E1, E2, E3, E4, I1, I2, I3, I4, R1, R2, R3, R4, S2, S3, S4 = state

        bb1 = self.get_beta(h)
        bb2 = 0.5*bb1
        bb3 = 0.35*bb1
        bb4 = 0.25*bb1

        I = I1 + I2 + I3 + I4
        R = R1 + R2 + R3 + R4

        dM = R*mu - (xi + mu)*M
        dS1 = mu*(1-R) + xi*M - mu*S1 - bb1*I*S1
        dE1 = bb1*I*S1 - (mu+sigma)*E1
        dE2 = bb2*I*S2 - (mu+sigma)*E2
        dE3 = bb3*I*S3 - (mu+sigma)*E3
        dE4 = bb4*I*S4 - (mu+sigma)*E4
        dI1 = sigma*E1 - (nu+mu)*I1
        dI2 = sigma*E2 - (nu+mu)*I2
        dI3 = sigma*E3 - (nu+mu)*I3
        dI4 = sigma*E4 - (nu+mu)*I4
        dR1 = nu*I1 - (mu+gamma)*R1
        dR2 = nu*I2 - (mu+gamma)*R2
        dR3 = nu*I3 - (mu+gamma)*R3
        dR4 = nu*I4 - (mu+gamma)*R4
        dS2 = gamma*R1 - mu*S2 - bb2*I*S2
        dS3 = gamma*R2 - mu*S3 - bb3*I*S3
        dS4 = gamma*(R3+R4) - mu*S4 - bb4*I*S4

        dstate = jnp.array([dM, dS1, dE1, dE2, dE3, dE4, dI1, dI2, dI3, dI4, dR1, dR2, dR3, dR4, dS2, dS3, dS4])

        dh = self.hidden_dyn(t, jnp.concatenate([h, state]), args)

        return (dstate, dh)

    def __call__(self, ts):
        y0 = jnn.softmax(self.y0)
        h0 = self.hidden_vec

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            # diffrax.Kvaerno5(),
            # diffrax.Dopri8(),
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

        return states, h  # normalized 상태로 반환


########## Experiment ##########

class Experiment(BaseExperiment):

    def __init__(self, ts, ys, hidden_dim=8, width_size=64, depth=4, **kwargs):

        seed = kwargs.get("seed", 5678)

        model = Argphy(
            hidden_dim,
            width_size,
            depth,
            key=jax.random.PRNGKey(seed),
        )

        super().__init__(model, ts, ys, **kwargs)

    def loss_fn(self, model, ts, ys):
        pred, _ = model(ts)
        I_pred = (pred[:,6] + pred[:,7] + pred[:,8] + pred[:,9]) * model.s
        loss = jnp.mean(jnp.square(I_pred - ys) / jnp.max(ys).squeeze())
        return loss


########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    ts_data, ys_data, model = EX.ts, EX.ys, EX.model

    ys_pred, h_pred = model(ts_eval)

    ys_pred = (ys_pred[:,6] + ys_pred[:,7] + ys_pred[:,8] + ys_pred[:,9]) * model.s

    beta_pred = jax.vmap(model.get_beta)(h_pred)

    plotting(ts_data, ys_data, ts_eval, ys_pred, beta_pred, loss_list, viz_data)