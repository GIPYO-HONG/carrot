import jax.numpy as jnp

import matplotlib.pyplot as plt

def relative_error(pred, true):
    return jnp.linalg.norm(pred - true, 2) / jnp.linalg.norm(true, 2)

def plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, eta_eval, eta_pred, loss_list, viz_data):
    rel_Tu = relative_error(ys_pred[:, 0], ys_eval[:, 0])
    rel_Ti = relative_error(ys_pred[:, 1], ys_eval[:, 1])
    rel_V = relative_error(ys_pred[:, 2], ys_eval[:, 2])
    rel_ee = relative_error(eta_pred, eta_eval)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].plot(ts_eval, ys_eval[:, 0], label="True T_U")
    axs[0].plot(ts_eval, ys_pred[:, 0], "--", label="Pred T_U")
    axs[0].set_title(f"S (Rel. Error: {rel_Tu:.2e})")
    axs[0].legend()

    axs[1].plot(ts_eval, ys_eval[:, 1], label="True T_I")
    axs[1].plot(ts_eval, ys_pred[:, 1], "--", label="Pred T_I")
    axs[1].set_title(f"E (Rel. Error: {rel_Ti:.2e})")
    axs[1].legend()

    axs[2].plot(ts_eval, ys_eval[:, 2], label="True V")
    axs[2].plot(ts_eval, ys_pred[:, 2], "--", label="Pred V")
    axs[2].set_title(f"I (Rel. Error: {rel_V:.2e})")
    axs[2].set_yscale("log")
    axs[2].legend()

    axs[3].plot(loss_list)
    axs[3].set_yscale("log")
    axs[3].set_title("Training Loss")

    plt.figure(figsize=(5, 5))
    plt.plot(ts_eval, eta_eval, label="True eta")
    plt.plot(ts_eval, eta_pred, label="Pred eta", linestyle="--")
    plt.title(f"beta (Rel. Error 1: {rel_ee:.2e})")
    plt.legend()
    plt.show()