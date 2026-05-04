import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxopt import LBFGS
import time
import os
import numpy as np
from tqdm import tqdm
from .logger import make_logger
from .optim_list import *


class BaseExperiment:
    """
    Base training experiment class.
    """

    def __init__(self, model, ts, ys, exp_name, base_dir="results", seed=5678):
        # -------- path define --------
        self.exp_dir = os.path.join(base_dir, exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "model_parameter")
        self.log_path = os.path.join(self.exp_dir, "train.log")
        self.loss_path = os.path.join(self.exp_dir, "loss_list.npy")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        # -----------------------------

        self.logger = make_logger(exp_name, self.log_path)

        self.model = model
        self.ts = ts
        self.ys = ys
        self.loss_list = []

    def save_model(self, step):
        path = os.path.join(self.ckpt_dir, f"model_step_{step:05d}.eqx")
        eqx.tree_serialise_leaves(path, self.model)

    def loss_fn(self, model, ts, ys):
        """
        You must define loss function in each model code.
        """
        raise NotImplementedError("loss function must return (loss, aux_data)")

    # ------------------------------------------------------------------ #
    #  Phase 1: Adam (or any optax optimizer)                             #
    # ------------------------------------------------------------------ #
    def train(self, optimizer=adam, lr=1e-3, steps=10000, viz_loss=1000):
        params, static = eqx.partition(self.model, eqx.is_inexact_array)

        optim = optimizer(lr)
        opt_state = optim.init(params)

        ts, ys = self.ts, self.ys

        @eqx.filter_value_and_grad
        def grad_loss(params, static, ts, ys):
            model = eqx.combine(params, static)
            return self.loss_fn(model, ts, ys)

        def step_fn(carry, _):
            params, opt_state = carry
            loss, grads = grad_loss(params, static, ts, ys)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            return (params, opt_state), loss

        @eqx.filter_jit
        def train_scan(params, opt_state, n_steps):
            (params, opt_state), losses = jax.lax.scan(
                step_fn, (params, opt_state), None, length=n_steps
            )
            return params, opt_state, losses

        self.logger.info(f"[Adam] start: {steps} steps, lr={lr}")
        train_start = time.time()
        num_cycles = steps // viz_loss

        for cycle in tqdm(range(num_cycles), desc="Adam", ncols=100):
            t0 = time.time()
            params, opt_state, batch_losses = train_scan(params, opt_state, viz_loss)

            self.loss_list.extend(batch_losses.tolist())
            self.model = eqx.combine(params, static)

            current_step = (cycle + 1) * viz_loss
            self.save_model(current_step)

            t1 = time.time()
            msg = (f"[Adam] step: {current_step:5d}, "
                   f"loss: {batch_losses[-1]:.6e}, time: {t1-t0:.2f}s")
            tqdm.write(msg)
            self.logger.info(msg)

        total_time = time.time() - train_start
        self.logger.info(f"[Adam] Total time: {total_time/60:.2f} min")
        np.save(self.loss_path, np.array(self.loss_list))
        self.model = eqx.combine(params, static)

    # ------------------------------------------------------------------ #
    #  Phase 2: LBFGS fine-tuning (jaxopt — Wolfe line search 내장)      #
    # ------------------------------------------------------------------ #
    def train_lbfgs(self, maxiter=2000, viz_loss=100, tol=1e-9):
        params, static = eqx.partition(self.model, eqx.is_inexact_array)
        ts, ys = self.ts, self.ys

        def loss_for_lbfgs(params):
            model = eqx.combine(params, static)
            return self.loss_fn(model, ts, ys)

        solver = LBFGS(
            fun=loss_for_lbfgs,
            maxiter=maxiter,
            tol=tol,
            jit=True,
        )

        self.logger.info(f"[LBFGS] start: {maxiter} iters, tol={tol}")
        train_start = time.time()
        adam_steps_done = len(self.loss_list)
        num_cycles = maxiter // viz_loss

        state = solver.init_state(params)
        update_jit = jax.jit(solver.update)

        for cycle in tqdm(range(num_cycles), desc="LBFGS", ncols=100):
            t0 = time.time()

            for _ in range(viz_loss):
                params, state = update_jit(params, state)

            global_iter = (cycle + 1) * viz_loss
            loss_val = float(state.value)
            self.loss_list.append(loss_val)
            self.model = eqx.combine(params, static)

            current_iter = adam_steps_done + global_iter
            self.save_model(current_iter)

            t1 = time.time()
            msg = (f"[LBFGS] iter: {global_iter:5d}/{maxiter}, "
                   f"loss: {loss_val:.6e}, "
                   f"grad_norm: {float(state.error):.2e}, "
                   f"time: {t1-t0:.2f}s")
            tqdm.write(msg)
            self.logger.info(msg)

            if state.error < tol:
                self.logger.info(f"[LBFGS] Converged at iter {global_iter}.")
                break

        total_time = time.time() - train_start
        self.logger.info(f"[LBFGS] Total time: {total_time/60:.2f} min")
        np.save(self.loss_path, np.array(self.loss_list))
        self.model = eqx.combine(params, static)