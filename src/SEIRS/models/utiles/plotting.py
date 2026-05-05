import jax.numpy as jnp

import matplotlib.pyplot as plt

def relative_error(pred, true):
    return jnp.linalg.norm(pred - true, 2) / jnp.linalg.norm(true, 2)

def plotting(ts_data, ys_data, ts_eval, ys_pred, beta_pred, loss_list, viz_data):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0][0].plot(ts_eval, ys_pred[:, 0], "--", label="Pred S")
    axs[0][0].legend()

    axs[0][1].plot(ts_eval, ys_pred[:, 1], "--", label="Pred E")
    axs[0][1].legend()

    axs[0][2].plot(ts_data, ys_data, ".", label="True I")
    axs[0][2].plot(ts_eval, ys_pred[:, 2], "--", label="Pred I")
    axs[0][2].legend()

    axs[1][0].plot(ts_eval, ys_pred[:, 3], "--", label="Pred R")
    axs[1][0].legend()

    axs[1][1].plot(ts_eval, jnp.sum(ys_pred, axis=1), "--", label="Pred N")
    axs[1][1].legend()

    axs[1][2].plot(loss_list)
    axs[1][2].set_yscale("log")
    axs[1][2].set_title("Training Loss")

    if viz_data == True:
        axs[0][0].plot(ts_data, ys_data[:,0], label="data")
        axs[0][1].plot(ts_data, ys_data[:,1], label="data")
        axs[0][2].plot(ts_data, ys_data[:,2], label="data")
        axs[1][0].plot(ts_data, ys_data[:,3], label="data")
        axs[1][1].plot(ts_data, ys_data[:,4], label="data")

    plt.figure(figsize=(5, 5))
    plt.plot(ts_eval, beta_pred, label="Pred beta", linestyle="--")
    plt.legend()
    plt.show()