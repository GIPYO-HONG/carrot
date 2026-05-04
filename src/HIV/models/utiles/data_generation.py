import jax.numpy as jnp

import diffrax

########## Data generation ##########
class beta_generate:
    def __init__(self, b0, b1, shift=0.):
        self.b0 = b0
        self.b1 = b1
        self.shift = shift
    
    def func(self, t):
        return self.b0*(1 + self.b1*jnp.cos(2*jnp.pi*((t/365) - self.shift)))
    
def SEIAR(t, y, beta):

    kk = 0.526
    aa = 0.244
    ii = 0.244
    p = 0.667
    f = 0.98
    ee = 0
    dd = 1
    q = 0.5

    S, E, I, A, R = y
    bb = beta(t) if callable(beta) else beta

    LL = ee*E + (1-q)*I + dd*A

    dS = -bb*S*LL
    dE = bb*S*LL - kk*E
    dI = p*kk*E - aa*I
    dA = (1-p)*kk*E - ii*A
    dR = f*aa*I + ii*A

    return jnp.array([dS, dE, dI, dA, dR])

def get_data(ts, y0, beta):

    def SEIAR_(t, y, args=None):
        return SEIAR(t, y, beta)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(SEIAR_),
        diffrax.Tsit5(),
        t0=0,
        t1=ts[-1],
        dt0=0.01,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        args=None,
        max_steps=500000
        )
    
    return sol.ys