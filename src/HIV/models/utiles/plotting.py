import jax.numpy as jnp

import matplotlib.pyplot as plt

def relative_error(pred, true):
    return jnp.linalg.norm(pred - true, 2) / jnp.linalg.norm(true, 2)

def plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, beta_eval, beta_pred, loss_list, viz_data):
    rel_S = relative_error(ys_pred[:, 0], ys_eval[:, 0])
    rel_E = relative_error(ys_pred[:, 1], ys_eval[:, 1])
    rel_I = relative_error(ys_pred[:, 2], ys_eval[:, 2])
    rel_A = relative_error(ys_pred[:, 3], ys_eval[:, 3])
    rel_R = relative_error(ys_pred[:, 4], ys_eval[:, 4])
    rel_bb = relative_error(beta_pred, beta_eval)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0][0].plot(ts_eval, ys_eval[:, 0], label="True S")
    axs[0][0].plot(ts_eval, ys_pred[:, 0], "--", label="Pred S")
    axs[0][0].set_title(f"S (Rel. Error: {rel_S:.2e})")
    axs[0][0].legend()

    axs[0][1].plot(ts_eval, ys_eval[:, 1], label="True E")
    axs[0][1].plot(ts_eval, ys_pred[:, 1], "--", label="Pred E")
    axs[0][1].set_title(f"E (Rel. Error: {rel_E:.2e})")
    axs[0][1].legend()

    axs[0][2].plot(ts_eval, ys_eval[:, 2], label="True I")
    axs[0][2].plot(ts_eval, ys_pred[:, 2], "--", label="Pred I")
    axs[0][2].set_title(f"I (Rel. Error: {rel_I:.2e})")
    axs[0][2].legend()

    axs[1][0].plot(ts_eval, ys_eval[:, 3], label="True A")
    axs[1][0].plot(ts_eval, ys_pred[:, 3], "--", label="Pred A")
    axs[1][0].set_title(f"A (Rel. Error: {rel_A:.2e})")
    axs[1][0].legend()

    axs[1][1].plot(ts_eval, ys_eval[:, 4], label="True R")
    axs[1][1].plot(ts_eval, ys_pred[:, 4], "--", label="Pred R")
    axs[1][1].set_title(f"R (Rel. Error: {rel_R:.2e})")
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
    plt.plot(ts_eval, beta_eval, label="True beta")
    plt.plot(ts_eval, beta_pred, label="Pred beta", linestyle="--")
    plt.title(f"beta (Rel. Error 1: {rel_bb:.2e})")
    plt.legend()
    plt.show()