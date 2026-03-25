import jax.numpy as jnp

from models import *
from models.utiles.data_generation import beta_generate, get_data
from models.utiles.optim_list import *

exp_name = "beta_argument_state"

model = ap2

beta = beta_generate(5e-1, 0.1, 0.)
y0 = jnp.array([1e+0, 0., 1e-6, 0., 0.])
ts = jnp.linspace(0., 365., 365+1)
ys = get_data(ts, y0, beta.func)

ts_eval = jnp.linspace(0., 365., 365*4+1)

EX = model.Experiment(
    ts=ts, 
    ys=ys,
    beta=beta.func, 
    exp_name=exp_name,
)

steps = 50000

if __name__=="__main__":
    EX.train(lr=1e-3, steps=steps)
