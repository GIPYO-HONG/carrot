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
            final_activation=lambda x: jnn.tanh(0.0001*x),
            key=key,
        )

    def __call__(self, t, h, args=None):
        return self.scale * self.mlp(h)


class Argphy(eqx.Module):
    hidden_dyn: Dynamics
    hidden_vec: jnp.ndarray
    hidden_to_eta: eqx.nn.Linear
    norm_scale: tuple = eqx.field(static=True)

    parameter: jnp.ndarray

    def __init__(self, hidden_dim, width_size, depth, norm_scale, *, key):
        dyn_key, htb_key, hvec_key = jr.split(key, 3)

        self.hidden_dyn = Dynamics(hidden_dim, width_size, depth, key=dyn_key)
        self.hidden_vec = 0.01 * jr.normal(hvec_key, (hidden_dim,))
        self.hidden_to_eta = eqx.nn.Linear(hidden_dim, 1, key=htb_key)
        self.norm_scale = tuple(norm_scale.tolist())

        self.parameter = jnp.array([20., 1., 1100., 1., 1.])

    def get_eta(self, h):
        eta = jnn.sigmoid(self.hidden_to_eta(h))
        return eta.squeeze()

    def RHS(self, t, y, args=None):
        norm_state, h = y

        scale = jnp.array(self.norm_scale)
        state = norm_state * scale
        Tu, Ti, V = state

        ee = self.get_eta(h)

        ll, rr, N, dd, c = jnn.softplus(self.parameter)

        dTu = ll - rr * Tu - ee * Tu * V
        dTi = ee * Tu * V - dd * Ti
        dV  = N * dd * Ti - c * V

        dstate = jnp.array([dTu, dTi, dV])
        dnorm_state = dstate / scale

        dh = self.hidden_dyn(t, jnp.concatenate([h, norm_state]), args)

        return (dnorm_state, dh)

    def __call__(self, y0, ts):
        h0 = self.hidden_vec

        scale = jnp.array(self.norm_scale)
        norm_y0 = y0 / scale

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            # diffrax.Tsit5(),
            diffrax.Kvaerno5(),
            # diffrax.Dopri8(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=(norm_y0, h0),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=50000,
        )

        norm_states, h = sol.ys

        return norm_states, h  # normalized 상태로 반환


########## Experiment ##########

class Experiment(BaseExperiment):

    def __init__(self, ts, y0, ys, eta, hidden_dim=8, width_size=64, depth=4, **kwargs):

        seed = kwargs.get("seed", 5678)

        scales_obs = jnp.max(ys, axis=0) + 1e-6  # shape: (2,), [T_max, V_max]
        self.scales = scales_obs

        # Argphy에는 state (TU, TI, V) 3개 스케일 전달
        self.norm_scale = jnp.array([scales_obs[0], scales_obs[0], scales_obs[1]])

        model = Argphy(
            hidden_dim,
            width_size,
            depth,
            norm_scale=self.norm_scale,
            key=jax.random.PRNGKey(seed),
        )

        super().__init__(model, ts, ys, **kwargs)

        self.y0  = y0
        self.eta = eta

    def loss_fn(self, model, ts, ys):
        pred, _ = model(self.y0, ts)  # normalized, shape: (T, 3)

        # scale[0] == scale[1]이므로 pred[:,0]+pred[:,1] = (TU+TI)/T_max
        T_pred = pred[:, 0] + pred[:, 1]
        V_pred = pred[:, 2]

        T_target = ys[:, 0] / self.scales[0]
        V_target = ys[:, 1] / self.scales[1]

        loss = jnp.mean(jnp.square(T_pred - T_target)) \
             + jnp.mean(jnp.square(V_pred - V_target))
        return loss


########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    y0, ts_data, ys_data, eta, model, scales = EX.y0, EX.ts, EX.ys, EX.eta, EX.model, EX.norm_scale

    ys_eval  = get_data(ts_eval, y0, eta)
    ys_pred, h_pred = model(y0, ts_eval)
    ys_pred = ys_pred * scales              # 복원해서 plotting

    eta_eval = EX.eta(ts_eval)
    eta_pred = jax.vmap(model.get_eta)(h_pred)

    plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, eta_eval, eta_pred, loss_list, viz_data)