import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax

########## Data generation ##########
def eta(t):
    return 9*10**(-5)*(1-0.9*jnp.cos(jnp.pi*t / 1000))
    
def HIV(t, y, eta):

    ll = 36
    rr = 0.108
    N = 1000
    dd = 0.5
    c = 3

    Tu, Ti, V = y
    ee = eta(t) if callable(eta) else eta

    dTu = ll - rr*Tu - ee*Tu*V
    dTi = ee*Tu*V - dd*Ti
    dV = N*dd*Ti - c*V

    return jnp.array([dTu, dTi, dV])

def get_data(ts, y0, eta):

    def SEIAR_(t, y, args=None):
        return HIV(t, y, eta)

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

if __name__ == "__main__":
    y0 = jnp.array([600., 30., 10**5])

    ts = jnp.linspace(0., 20., 20*4+1)
    ys = get_data(ts, y0, eta)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(ts, ys[:, 0], label="T_U")
    axs[0].legend()

    axs[1].plot(ts, ys[:, 1], label="T_I")
    axs[1].legend()

    axs[2].plot(ts, ys[:, 2], label="V")
    axs[2].set_yscale('log')
    axs[2].legend()

    plt.tight_layout()
    plt.show()