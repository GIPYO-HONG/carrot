import jax.numpy as jnp
import matplotlib.pyplot as plt

import diffrax

########## Data generation ##########
class beta_generate:
    def __init__(self, b0, b1, phi, scale):
        self.b0 = b0
        self.b1 = b1
        self.phi = phi
        self.scale = scale
    
    def func(self, t):
        return (self.b0 / self.scale)*(1 + self.b1*jnp.cos(2*jnp.pi*(t / self.scale + self.phi)))
    
def MSEIRS4(t, y, beta, scale):

    xi, mu, sigma, nu, gamma = 13 / scale, 0.041 / scale, 91 / scale, 36 / scale, 1.8 / scale

    M, S1, E1, E2, E3, E4, I1, I2, I3, I4, R1, R2, R3, R4, S2, S3, S4 = y

    bb1 = beta(t) if callable(beta) else beta
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

    return jnp.array([dM, dS1, dE1, dE2, dE3, dE4, dI1, dI2, dI3, dI4, dR1, dR2, dR3, dR4, dS2, dS3, dS4])

def get_data(ts, y0, beta, scale):

    def MSEIRS4_(t, y, args=None):
        return MSEIRS4(t, y, beta, scale)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(MSEIRS4_),
        diffrax.Tsit5(),
        t0=0,
        t1=ts[-1],
        dt0=0.001,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        args=None,
        max_steps=500000
        )
    
    return sol.ys

if __name__ == "__main__":
    beta = beta_generate(200, 0.1, 0., scale=1).func
    y0 = jnp.full((17,), 1.0/17.0)
    ts = jnp.linspace(0, 100, 100)

    ys = get_data(ts, y0, beta, scale=1)

    print(ys[-1,:])

    beta_eval = beta_generate(200, 0.1, 0., scale=12).func
    y0_eval = ys[-1,:]
    ts_eval = jnp.linspace(0, 12*4, 12*4+1)
    ys_eval = get_data(ts_eval, y0_eval, beta_eval, scale=12)

    plt.figure(figsize=(5, 5))
    plt.plot(ts_eval, ys_eval[:,6]+ys_eval[:,7]+ys_eval[:,8]+ys_eval[:,9])
    plt.show()
