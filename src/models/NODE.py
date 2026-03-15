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

class Func(eqx.Module):
    beta_net: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):
        self.beta_net = eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.tanh(x),
            final_activation=lambda x: 3 * jnn.sigmoid(x),
            key=key
            )
        
    def __call__(self, t, y, args=None):
        S, E, I, A, R = y
        t_input = jnp.expand_dims(t, axis=0)
        beta = self.beta_net(t_input)[0]

        kk = 0.526
        aa = 0.244
        ii = 0.244
        p = 0.667
        f = 0.98
        ee = 0.0
        dd = 1.0
        q = 0.5

        LL = ee*E + (1-q)*I + dd*A
        dS = -beta*S*LL
        dE = beta*S*LL - kk*E
        dI = p*kk*E - aa*I
        dA = (1-p)*kk*E - ii*A
        dR = f*aa*I + ii*A

        return jnp.array([dS, dE, dI, dA, dR])
    
class NODE(eqx.Module):
    func: Func

    def __init__(self, width_size, depth, *, key):
        self.func = Func(width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.01,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6, dtmin=1e-3),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=100000,
        )
        return solution.ys

    
########## Experiment ##########
    
class Experiment(BaseExperiment):
    def __init__(self, beta, y0, ts, ys, I_only=False, width_size=64, depth=2, **kwargs):
        seed = kwargs.get('seed', 5678)
        model = NODE(width_size, depth, key=jr.PRNGKey(seed))
        
        super().__init__(model, y0, ts, ys, **kwargs)
        
        self.I_only = I_only
        self.beta = beta
        
        self.y0 = y0

        self.scales = jnp.max(ys, axis=0) + 1e-4

    def loss_fn(self, model, ts, ys):
        pred = model(ts, self.y0)
        
        if self.I_only:
            loss = jnp.mean(jnp.square((pred[:, 2] - ys[:, 2]) / self.scales[2]))
        else:
            loss = jnp.mean(jnp.square((pred - ys) / self.scales))
            
        return loss, None
    
########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list, viz_data=False):
    y0, ts_data, ys_data, model, I_only = EX.y0, EX.ts_data, EX.ys_data, EX.model, EX.I_only
    ys_eval = get_data(ts_eval, y0, EX.beta)
    ys_pred = model(ts_eval, y0)
    beta_eval = EX.beta(ts_eval)
    beta_pred = jax.vmap(lambda t: model.func.beta_net(jnp.array([t]))[0])(ts_eval)

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