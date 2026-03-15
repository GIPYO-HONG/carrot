import jax
jax.config.update("jax_debug_nans", True)

import jax.numpy as jnp

from models import *
from models.utiles.data_generation import beta_generate, get_data
from models.utiles.optim_list import *
from optax import adamw, adam

exp_name = "tmp_exp"

model = tmp

beta = beta_generate(5e-1, 0.1, 0.)
y0 = jnp.array([1e+0, 0., 1e-6, 0., 0.])
ts = jnp.linspace(0., 365., 365+1)
ys = get_data(ts, y0, beta.func)
# ys = jnp.expand_dims(ys[:,2], axis=1)

# 실험 편의를 위해 pinn의 collocation points를 0부터 365일에서 365*2+1개로 고정
I_only = False
EX = model.Experiment(
    ts=ts, 
    ys=ys,
    beta=beta.func, 
    exp_name=exp_name,
)

steps = 10000

if __name__=="__main__":
    EX.train(optimizer= adamw, lr=1e-3, steps=steps)
