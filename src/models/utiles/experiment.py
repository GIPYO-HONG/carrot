# import jax
# import equinox as eqx
# import optax
# import time
# import os
# import numpy as np
# from tqdm import tqdm
# from .logger import make_logger

# class BaseExperiment:
#     def __init__(self, model, y0, ts, ys, exp_name, base_dir="results", seed=5678):
#         self.exp_dir = os.path.join(base_dir, exp_name)
#         self.ckpt_dir = os.path.join(self.exp_dir, "model_parameter")
#         self.log_path = os.path.join(self.exp_dir, "train.log")
#         self.loss_path = os.path.join(self.exp_dir, "loss_list.npy")
        
#         os.makedirs(self.ckpt_dir, exist_ok=True)
#         self.logger = make_logger(exp_name, self.log_path)

#         self.model = model
#         self.y0 = y0
#         self.ts_data = ts
#         self.ys_data = ys
#         self.loss_list = []
#         self.aux_list = []

#     def save_model(self, step):
#         path = os.path.join(self.ckpt_dir, f"model_step_{step:05d}.eqx")
#         eqx.tree_serialise_leaves(path, self.model)

#     def calculate_loss(self, model, ts, ys):
#         raise NotImplementedError("calculate_loss must return (loss, aux_data)")

#     def train(self, lr=1e-3, steps=10000, print_every=1000):
#         optim = optax.adam(lr)
#         diff_model, static_model = eqx.partition(self.model, eqx.is_inexact_array)
#         opt_state = optim.init(diff_model)

#         ts_data, ys_data = self.ts_data, self.ys_data
#         calc_loss_fn = self.calculate_loss 

#         @eqx.filter_value_and_grad(has_aux=True)
#         def grad_loss(d_model, s_model, ts, ys):
#             model = eqx.combine(d_model, s_model)
#             return calc_loss_fn(model, ts, ys)

#         def step_fn(carry, _):
#             d_model, o_state = carry
#             (loss, aux), grads = grad_loss(d_model, static_model, ts_data, ys_data)
#             updates, o_state = optim.update(grads, o_state, d_model)
#             d_model = eqx.apply_updates(d_model, updates)
#             return (d_model, o_state), (loss, aux)

#         @eqx.filter_jit
#         def train_scan(d_model, o_state, n_steps):
#             init_val = (d_model, o_state)
#             (d_model, o_state), (losses, aux_collection) = jax.lax.scan(
#                 step_fn, init_val, None, length=n_steps
#             )
#             return d_model, o_state, losses, aux_collection

#         curr_diff_model = diff_model
#         num_cycles = steps // print_every
        
#         self.logger.info(f"Start Training: {steps} steps")
#         train_start = time.time()

#         for cycle in tqdm(range(num_cycles), desc="Training", ncols=100):
#             t0 = time.time()
            
#             curr_diff_model, opt_state, batch_losses, batch_aux = train_scan(
#                 curr_diff_model, opt_state, print_every
#             )
            
#             self.loss_list.extend(batch_losses.tolist())
#             self.aux_list = batch_aux 

#             self.model = eqx.combine(curr_diff_model, static_model)
            
#             current_step = (cycle + 1) * print_every
#             self.save_model(current_step)

#             t1 = time.time()
#             msg = f"step: {current_step:5d}, loss: {batch_losses[-1]:.6e}, time: {t1 - t0:.2f}s"
#             tqdm.write(msg)
#             # print(msg)
#             self.logger.info(msg)

#         total_time = time.time() - train_start
#         self.logger.info(f"Total time: {total_time/60:.2f} min")
#         np.save(self.loss_path, np.array(self.loss_list))

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
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

    def __init__(self, model, y0, ts, ys, exp_name, base_dir="results", seed=5678):
        #-------- path define --------
        self.exp_dir = os.path.join(base_dir, exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "model_parameter")
        self.log_path = os.path.join(self.exp_dir, "train.log")
        self.loss_path = os.path.join(self.exp_dir, "loss_list.npy")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        #-----------------------------

        # save training log
        self.logger = make_logger(exp_name, self.log_path)

        # save data
        self.model = model
        self.y0 = y0
        self.ts = ts
        self.ys = ys
        self.loss_list = []
        self.aux_list = []

    # save model checkpoint
    def save_model(self, step):
        path = os.path.join(self.ckpt_dir, f"model_step_{step:05d}.eqx")
        eqx.tree_serialise_leaves(path, self.model)

    def loss_fn(self, model, ts, ys):
        """
        You must define loss function in each model code.
        """
        raise NotImplementedError("loss function must return (loss, aux_data)")
    
    # training loop
    def train(self, optimizer=adam, lr=1e-3, steps=10000, viz_loss=1000):
        params, static = eqx.partition(self.model, eqx.is_inexact_array)
        
        optim = optimizer(lr)
        opt_state = optim.init(params)
        
        ts, ys = self.ts, self.ys

        # output : (loss,aux), grad
        @eqx.filter_value_and_grad(has_aux=True)
        def grad_loss(params, static, ts, ys):
            model = eqx.combine(params, static)
            return self.loss_fn(model, ts, ys)
        
        # perform 1 epoch
        def step_fn(carry, _):
            params, opt_state = carry
            (loss, aux), grads = grad_loss(params, static, ts, ys)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            return (params, opt_state), (loss, aux)
        
        # Use scan for fast training
        @eqx.filter_jit
        def train_scan(params, opt_state, n_steps):
            init_val = (params, opt_state)
            (params, opt_state), (losses, aux_collection) = jax.lax.scan(
                step_fn, init_val, None, length=n_steps
            )
            return params, opt_state, losses, aux_collection
        
        num_cycles = steps // viz_loss #for visualize loss decay

        #-------- Training start --------
        self.logger.info(f"start Training: {steps} steps")

        train_start = time.time()

        for cycle in tqdm(range(num_cycles), desc="Training", ncols=100):
            t0 = time.time()

            params, opt_state, batch_losses, batch_aux = train_scan(
                params, opt_state, viz_loss
            )

            self.loss_list.append(batch_losses)
            self.aux_list.append(batch_aux)

            self.model = eqx.combine(params, static)

            current_step = (cycle+1)*viz_loss
            self.save_model(current_step)

            t1 = time.time()
            msg = f"step: {current_step:5d}, loss: {batch_losses[-1]:.6e}, time: {t1 - t0:.2f}s"
            tqdm.write(msg)
            self.logger.info(msg)

        total_time = time.time() - train_start
        self.logger.info(f"Total time: {total_time/60:.2f} min")

        loss_array = jnp.concatenate(self.loss_list)
        np.save(self.loss_path, np.array(loss_array))

        self.model = eqx.combine(params, static)

