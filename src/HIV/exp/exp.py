import jax.numpy as jnp

from models import *

exp_name = "exp"

model = an

y0 = jnp.array([600., 30., 10**5])
ts = jnp.linspace(0., 20., 200)
ys = get_data(ts, y0, eta)

ts_eval = jnp.linspace(0., 20., 800)

EX = model.Experiment(
    ts=ts,
    y0=y0,
    ys=ys,
    eta=eta, 
    exp_name=exp_name,
)

steps = 50000

if __name__=="__main__":
    EX.train(lr=1e-3, steps=steps)