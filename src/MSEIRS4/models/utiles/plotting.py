import jax.numpy as jnp

import matplotlib.pyplot as plt

def relative_error(pred, true):
    return jnp.linalg.norm(pred - true, 2) / jnp.linalg.norm(true, 2)

def plotting(ts_data, ys_data, ts_eval, ys_pred, beta_pred, loss_list, viz_data):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(ts_data, ys_data, ".", label="True I")
    axs[0].plot(ts_eval, ys_pred, "--", label="Pred I")
    axs[0].legend()

    axs[1].plot(loss_list)
    axs[1].set_yscale("log")
    axs[1].set_title("Training Loss")

    axs[2].plot(ts_eval, beta_pred, label="Pred beta")
    axs[2].legend()