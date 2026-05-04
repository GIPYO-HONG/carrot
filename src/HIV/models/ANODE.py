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
            in_size=hidden_dim + 3,
            out_size=hidden_dim,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.softplus(x),
            final_activation=lambda x: jnn.tanh(x),
            key=key,
        )

    def __call__(self, t, h, args=None):
        return self.scale * self.mlp(h)


class Argphy(eqx.Module):
    hidden_dyn: Dynamics
    hidden_vec: jnp.ndarray
    hidden_to_eta: eqx.nn.Linear
    norm_scale: tuple = eqx.field(static=True)

    def __init__(self, hidden_dim, width_size, depth, norm_scale, *, key):
        dyn_key, htb_key, hvec_key = jr.split(key, 3)

        self.hidden_dyn = Dynamics(hidden_dim, width_size, depth, key=dyn_key)
        self.hidden_vec = 0.01 * jr.normal(hvec_key, (hidden_dim,))
        self.hidden_to_eta = eqx.nn.Linear(hidden_dim, 1, key=htb_key)
        self.norm_scale = tuple(norm_scale.tolist())

    def get_eta(self, h):
        eta = jnn.sigmoid(self.hidden_to_eta(h))
        return eta.squeeze()

    def RHS(self, t, y, args=None):
        norm_state, h = y

        scale = jnp.array(self.norm_scale)
        state = norm_state * scale
        Tu, Ti, V = state

        ee = self.get_eta(h)

        ll = 36.0
        rr = 0.108
        N  = 1000.0
        dd = 0.5
        c  = 3.0

        dTu = ll - rr * Tu - ee * Tu * V
        dTi = ee * Tu * V - dd * Ti
        dV  = N * dd * Ti - c * V

        dstate = jnp.array([dTu, dTi, dV])
        dnorm_state = dstate / scale

        dh = self.hidden_dyn(t, jnp.concatenate([h, state]), args)

        return (dnorm_state, dh)

    def __call__(self, y0, ts):
        h0 = self.hidden_vec

        scale = jnp.array(self.norm_scale)
        norm_y0 = y0 / scale

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            # diffrax.Kvaerno5(),
            # diffrax.Dopri8(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=(norm_y0, h0),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-2, atol=1e-4),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=500000,
        )

        norm_states, h = sol.ys

        return norm_states, h  # normalized 상태로 반환


########## Experiment ##########

class Experiment(BaseExperiment):

    def __init__(self, ts, y0, ys, eta, hidden_dim=8, width_size=64, depth=4, **kwargs):

        seed = kwargs.get("seed", 5678)

        self.scales = jnp.max(ys, axis=0) + 1e-6

        model = Argphy(
            hidden_dim,
            width_size,
            depth,
            norm_scale=self.scales,
            key=jax.random.PRNGKey(seed),
        )

        super().__init__(model, ts, ys, **kwargs)

        self.y0  = y0
        self.eta = eta

    def loss_fn(self, model, ts, ys):
        ys_norm = ys / self.scales          # target도 normalize
        pred, _ = model(self.y0, ts)        # pred는 이미 normalized
        loss = jnp.mean(jnp.square(pred - ys_norm))  # ★ 괄호 수정
        return loss


########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    y0, ts_data, ys_data, eta, model, scales = EX.y0, EX.ts, EX.ys, EX.eta, EX.model, EX.scales

    ys_eval  = get_data(ts_eval, y0, eta)
    ys_pred, h_pred = model(y0, ts_eval)
    ys_pred = ys_pred * scales              # 복원해서 plotting

    eta_eval = EX.eta(ts_eval)
    eta_pred = jax.vmap(model.get_eta)(h_pred)

    plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, eta_eval, eta_pred, loss_list, viz_data)