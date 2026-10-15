import numpy as np
import torch as th
import math

from flow_matching.path.affine import AffineProbPath
from flow_matching.path.scheduler.scheduler import (
    CondOTScheduler,
    PolynomialConvexScheduler,
    VPScheduler,
    LinearVPScheduler,
    CosineScheduler,
)
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper, expand_tensor_like

from utils.batch_util import _split_data, _combine_data
from utils.feature_accessor import get_traj_features_from_batch
from model.metadata_moe import assert_action_blind_mapping, sum_loss_dict


def _create_scheduler(scheduler_type: str, scheduler_kwargs: dict | None = None):
    """
    Factory function to create a scheduler based on the scheduler type.
    
    Args:
        scheduler_type: Type of scheduler ('condot', 'polynomial', 'vp', 'linear_vp', 'cosine')
        scheduler_kwargs: Dictionary of keyword arguments for the scheduler
    
    Returns:
        Scheduler instance
    """
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    
    if scheduler_type == 'condot':
        return CondOTScheduler()
    elif scheduler_type == 'polynomial':
        n = scheduler_kwargs.get('polynomial_n', 1.0)
        return PolynomialConvexScheduler(n=n)
    elif scheduler_type == 'vp':
        beta_min = scheduler_kwargs.get('vp_beta_min', 0.1)
        beta_max = scheduler_kwargs.get('vp_beta_max', 20.0)
        return VPScheduler(beta_min=beta_min, beta_max=beta_max)
    elif scheduler_type == 'linear_vp':
        return LinearVPScheduler()
    elif scheduler_type == 'cosine':
        return CosineScheduler()
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                        f"Supported types: 'condot', 'polynomial', 'vp', 'linear_vp', 'cosine'")


class FlowMatchingModel:
    """
    Flow Matching implementation for the PoseNet motion generation task.
    This class is API-compatible with other generative models in the pose pipeline.
    """
    def __init__(self,
            num_total_steps,
            ode_solver,
            pred_target,
            discrete_timestep_sampling: bool,
            task,
            device,
            dataloader=None,
            # new sampler configuration (defaults chosen per paper)
            time_sampler: str = 'uniform',   # 'uniform' | 'logit-normal' | 'mode' | 'cosmap'
            logit_m: float = 0.0,
            logit_s: float = 1.0,
            mode_s: float = 1.29,
            eps_t: float = 5e-3,
            # scheduler configuration
            scheduler_type: str = 'condot',  # 'condot' | 'polynomial' | 'vp' | 'linear_vp' | 'cosine'
            scheduler_kwargs: dict | None = None,
            # ODE solver configuration
            ode_step_size: float | None = None,
            ode_time_grid_steps: int | None = None,
            sampling_noise_scale: float = 1.0,
            # Geometry/manifold regularization curriculum
            geometry_start_step: int = 50000,
            geometry_warmup_steps: int = 100000,
            geometry_max_weight: float = 1.0,
            geometry_time_weight_power: float = 2.0,
            geometry_time_weight_min_t: float = 0.0,
            geometry_time_weight_floor: float = 0.0,
            geometry_pred_x1_grad_scale: float = 1.0,
            geometry_level1_start_step: int | None = None,
            geometry_level1_warmup_steps: int | None = None,
            geometry_level2_start_step: int | None = None,
            geometry_level2_warmup_steps: int | None = None,
            geometry_level3_start_step: int | None = None,
            geometry_level3_warmup_steps: int | None = None,
            log_gradient_norms: bool = False,
            **kwargs):
        # kwargs absorb unused arguments from the factory like betas, etc.
        if dataloader is None:
            raise ValueError("FlowMatchingModel requires an explicit dataloader so it can read dataset metadata.")
        if scheduler_kwargs is None:
            raise ValueError("FlowMatchingModel requires explicit scheduler_kwargs in the Hydra config; use {} when the scheduler has no kwargs.")
        scheduler = _create_scheduler(scheduler_type, scheduler_kwargs)
        self.path = AffineProbPath(scheduler=scheduler)
        self.ode_solver = None
        self.ode_solver_name = ode_solver
        if hasattr(dataloader.dataset, "datasets"):
            self.ds = dataloader.dataset.datasets[0]
        else:
            self.ds = dataloader.dataset
        self.device = device
        self.task = task
        self.predicted_modalities = self.ds.repr_registry.predicted_modalities[self.task]
        self.pred_target = pred_target # 'x1', 'velocity', or 'epsilon'/'eps'

        self.discrete_timestep_sampling = discrete_timestep_sampling
        if self.discrete_timestep_sampling:
            self.num_total_steps = num_total_steps
            log2_steps = int(np.log2(self.num_total_steps))
            # possible_d_steps: [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]
            self.possible_d_steps = (2 ** np.arange(log2_steps)) / self.num_total_steps
        
        if ode_solver == 'discrete_euler':
            assert self.discrete_timestep_sampling, "When using 'discrete_euler' ode_solver, only discrete timestep sampling is supported."
        else:
            assert not self.discrete_timestep_sampling, "When using non-discrete timestep sampling, only 'discrete_euler' ode_solver is supported."
        
        # Time-sampling configuration (new)

        self.time_sampler = time_sampler
        self.logit_m = logit_m
        self.logit_s = logit_s
        self.mode_s = mode_s
        self.eps_t = eps_t
        
        # ODE solver configuration
        self.ode_step_size = ode_step_size
        self.ode_time_grid_steps = ode_time_grid_steps
        self.sampling_noise_scale = float(sampling_noise_scale)

        # Geometry losses regularize the predicted endpoint manifold. They are
        # delayed and time-weighted so they do not dominate early vector-field formation.
        self.geometry_start_step = int(geometry_start_step)
        self.geometry_warmup_steps = int(geometry_warmup_steps)
        self.geometry_max_weight = float(geometry_max_weight)
        self.geometry_time_weight_power = float(geometry_time_weight_power)
        self.geometry_time_weight_min_t = float(geometry_time_weight_min_t)
        self.geometry_time_weight_floor = float(geometry_time_weight_floor)
        self.geometry_pred_x1_grad_scale = float(geometry_pred_x1_grad_scale)
        self.geometry_level1_start_step = self.geometry_start_step if geometry_level1_start_step is None else int(geometry_level1_start_step)
        self.geometry_level1_warmup_steps = self.geometry_warmup_steps if geometry_level1_warmup_steps is None else int(geometry_level1_warmup_steps)
        self.geometry_level2_start_step = self.geometry_start_step if geometry_level2_start_step is None else int(geometry_level2_start_step)
        self.geometry_level2_warmup_steps = self.geometry_warmup_steps if geometry_level2_warmup_steps is None else int(geometry_level2_warmup_steps)
        self.geometry_level3_start_step = self.geometry_start_step if geometry_level3_start_step is None else int(geometry_level3_start_step)
        self.geometry_level3_warmup_steps = self.geometry_warmup_steps if geometry_level3_warmup_steps is None else int(geometry_level3_warmup_steps)
        self.log_gradient_norms = bool(log_gradient_norms)


    # --- add sampling helper methods inside the class ---

    def _sample_logit_normal(self, bs: int, device: th.device, m: float, s: float) -> th.Tensor:
        """
        logit(t) ~ Normal(m, s) -> t = sigmoid(u), u ~ N(m, s)
        returns t in (0,1)
        """
        u = th.randn(bs, device=device, dtype=th.float32) * float(s) + float(m)  # shape [bs]
        t = th.sigmoid(u)
        # clip to avoid exact 0 or 1
        return t.clamp(min=self.eps_t, max=1.0 - self.eps_t)

    def _sample_mode(self, bs: int, device: th.device, s: float) -> th.Tensor:
        """
        mode mapping fmode(u; s) as in the paper:
        fmode(u; s) = 1 - u - s * (cos^2(pi/2 u) - 1 + u)
        We sample u ~ Uniform(0,1) and set t = fmode(u; s).
        This follows the paper's construction where t = f(u) with u~U(0,1).
        """
        u = th.rand(bs, device=device, dtype=th.float32)
        # compute cos^2 term
        cos_term = th.cos((math.pi / 2.0) * u) ** 2
        t = 1.0 - u - float(s) * (cos_term - 1.0 + u)
        # clamp
        return t.clamp(min=self.eps_t, max=1.0 - self.eps_t)

    def _sample_cosmap(self, bs: int, device: th.device) -> th.Tensor:
        """
        CosMap mapping: solve for t given uniform u such that log-snr matches cosine schedule.
        paper gives: t = 1 - 1/(tan(pi/2 * u) + 1)
        with u ~ Uniform(0,1)
        """
        u = th.rand(bs, device=device, dtype=th.float32)
        # compute tan safely: avoid u very near 0.5 where tan finite but ok. also avoid exact pi/2.
        # numeric stability: clamp u away from 0 and 1.
        u_safe = u.clamp(min=self.eps_t, max=1.0 - self.eps_t)
        tan_part = th.tan((math.pi / 2.0) * u_safe)
        t = 1.0 - 1.0 / (tan_part + 1.0)
        return t.clamp(min=self.eps_t, max=1.0 - self.eps_t)

    def _sample_t(self, bs: int, device: th.device) -> th.Tensor:
        """
        Wrapper to sample t according to selected π(t).
        """
        if self.time_sampler == 'uniform':
            return th.rand(bs, device=device, dtype=th.float32).clamp(min=self.eps_t, max=1.0 - self.eps_t)
        elif self.time_sampler == 'logit-normal':
            return self._sample_logit_normal(bs, device, self.logit_m, self.logit_s)
        elif self.time_sampler == 'mode':
            return self._sample_mode(bs, device, self.mode_s)
        elif self.time_sampler == 'cosmap':
            return self._sample_cosmap(bs, device)
        else:
            # unknown sampler -> fallback to uniform and warn
            th._assert(False, f"Unknown time_sampler '{self.time_sampler}'", (self.time_sampler,))
            return th.rand(bs, device=device, dtype=th.float32).clamp(min=self.eps_t, max=1.0 - self.eps_t)

    def _sample_discrete_t(self, bs: int, device: th.device) -> th.Tensor:
        step_idx = th.randint(0, self.num_total_steps, (bs,), dtype=th.float32, device=device)
        return (step_idx / self.num_total_steps).clamp(min=self.eps_t, max=1.0 - self.eps_t)

    def _ramp(self, step: int, start_step: int, warmup_steps: int) -> float:
        if step < start_step:
            return 0.0
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step - start_step) / float(warmup_steps))

    def _geometry_time_weight(self, t_fm: th.Tensor) -> th.Tensor:
        w = t_fm.clamp(min=0.0, max=1.0)
        if self.geometry_time_weight_min_t > 0.0:
            w = th.where(w >= self.geometry_time_weight_min_t, w, th.zeros_like(w))
        if self.geometry_time_weight_power != 1.0:
            w = w.pow(self.geometry_time_weight_power)
        if self.geometry_time_weight_floor > 0.0:
            w = w.clamp(min=self.geometry_time_weight_floor)
        return w

    def _geometry_weights(self, t_fm: th.Tensor, current_step: int):
        time_weight = self._geometry_time_weight(t_fm)
        global_weight = self.geometry_max_weight * self._ramp(
            current_step,
            self.geometry_start_step,
            self.geometry_warmup_steps,
        )
        level_weights = {
            'level1': global_weight * self._ramp(current_step, self.geometry_level1_start_step, self.geometry_level1_warmup_steps),
            'level2': global_weight * self._ramp(current_step, self.geometry_level2_start_step, self.geometry_level2_warmup_steps),
            'level3': global_weight * self._ramp(current_step, self.geometry_level3_start_step, self.geometry_level3_warmup_steps),
        }
        return global_weight, time_weight, level_weights

    def _base_geometry_loss_weight(self, model, key: str) -> float:
        attr_by_key = {
            'loss_rec_repr_full_body': 'weight_loss_rec_repr_full_body',
            'loss_repr_foot_contact_mse': 'weight_loss_repr_foot_contact_mse',
            'loss_joint_pos_global': 'weight_loss_joint_pos_global',
            'loss_visible_joint_pos_global': 'weight_loss_visible_joint_pos_global',
            'loss_joint_vel_global': 'weight_loss_joint_vel_global',
            'loss_joint_smooth': 'weight_loss_joint_smooth',
            'loss_foot_skating': 'weight_loss_foot_skating',
            'loss_root_rec_repr': 'weight_loss_root_rec_repr',
            'loss_root_pos_global': 'weight_loss_root_pos_global',
            'loss_root_vel_global': 'weight_loss_root_vel_global',
            'loss_root_smooth': 'weight_loss_root_smooth',
            'loss_root_smplx_rot_vel': 'weight_loss_root_smplx_rot_vel',
            'loss_root_smplx_transl_vel': 'weight_loss_root_smplx_transl_vel',
            'loss_root_rot_vel_from_abs_traj': 'weight_loss_root_rot_vel_from_abs_traj',
            'loss_root_rot_cos_smooth_from_abs_traj': 'weight_loss_root_rot_cos_smooth_from_abs_traj',
        }
        attr = attr_by_key.get(key)
        return float(getattr(model, attr, 1.0)) if attr is not None else 1.0

    def _scale_loss_dict(self, loss_dict: dict, model, level_weights: dict, time_weight_mean: th.Tensor):
        level1_keys = {
            'loss_rec_repr_full_body',
            'loss_root_rec_repr',
            'loss_joint_pos_global',
            'loss_visible_joint_pos_global',
            'loss_root_pos_global',
            'loss_joint_vel_global',
            'loss_root_vel_global',
            'loss_root_smplx_rot_vel',
            'loss_root_smplx_transl_vel',
        }
        level2_keys = {
            'loss_repr_foot_contact_mse',
        }
        level3_keys = {
            'loss_foot_skating',
            'loss_joint_smooth',
            'loss_root_smooth',
            'loss_root_rot_cos_smooth_from_abs_traj',
            'loss_root_rot_vel_from_abs_traj',
        }
        loss_keys = level1_keys | level2_keys | level3_keys
        scaled = {}
        loss_terms = []
        for key, value in loss_dict.items():
            if key == 'loss':
                continue
            if not th.is_tensor(value):
                scaled[key] = value
                continue
            if key not in loss_keys:
                scaled[key] = value.detach()
                continue
            if key in level1_keys:
                weight = level_weights['level1']
            elif key in level2_keys:
                weight = level_weights['level2']
            else:
                weight = level_weights['level3']
            scaled_value = value * self._base_geometry_loss_weight(model, key) * weight * time_weight_mean
            scaled[key] = scaled_value
            loss_terms.append(scaled_value)
        if loss_terms:
            geom_loss = loss_terms[0].new_zeros(())
            for value in loss_terms:
                geom_loss = geom_loss + value
            scaled['loss'] = geom_loss
        else:
            ref = next((value for value in loss_dict.values() if th.is_tensor(value)), None)
            if ref is not None:
                scaled['loss'] = ref.new_zeros(())
        return scaled

    def _tensor_grad_norm(self, loss: th.Tensor, tensor: th.Tensor) -> th.Tensor:
        if not loss.requires_grad or not tensor.requires_grad:
            return tensor.new_zeros(())
        grad = th.autograd.grad(loss, tensor, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            return tensor.new_zeros(())
        return grad.detach().pow(2).sum().sqrt()

    def _slice_batch(self, batch: dict, sample_mask: th.Tensor) -> dict:
        bs = sample_mask.shape[0]
        return {
            key: value[sample_mask] if th.is_tensor(value) and value.shape[:1] == (bs,) else value
            for key, value in batch.items()
        }

    def training_losses(self, model, batch, t, noise=None, smplx_model=None, epoch=0, **kwargs):
        """
        Computes the Flow Matching training loss for PoseNet.
        't' and 'noise' from the training loop are ignored.
        """
        # THOUGHT: flow matching (and other deep generative models) models the original dataset distribution, so have to sample from dataset D and tries to recover original samples.
        # When using key as 'motion_repr_noisy', we sample from batch['motion_repr_noisy'] and tries to recover batch['motion_repr_clean'], which might break the formulation of flow matching.
        # Additionally, we cannot sample from batch['motion_repr_noisy'] and recover batch['motion_repr_noisy'] since this would lead to oscillation and noise in output features.
        # Therefore, the choice is to sample from batch['motion_repr_clean'] currently.
        full_key = 'motion_repr_clean'
        x1_full = batch[full_key]
        bs, device = x1_full.shape[0], x1_full.device
        _unwrap_model = model.module if hasattr(model, 'module') else model

        x1_gt, x1_modeled = _split_data(
            motion_repr=x1_full,
            task=self.task,
            ds=self.ds,
            predicted_modalities=self.predicted_modalities,
        )

        # Sample from the probability path p_t(x|x_0, x_1).
        x0 = th.randn_like(x1_modeled)

        if self.discrete_timestep_sampling:
            t_fm = self._sample_discrete_t(bs, device)
        else:
            # continuous-time sampling using selected π(t)
            t_fm = self._sample_t(bs, device)

        path_sample = self.path.sample(x_0=x0, x_1=x1_modeled, t=t_fm)
        xt = path_sample.x_t
        dxt = path_sample.dx_t

        batch['x_t'] = xt
        model_pred = model(batch, t_fm)

        if not self.discrete_timestep_sampling and self.time_sampler == 'uniform':

            bs = x1_modeled.shape[0]

            # choose target tensor depending on pred_target
            if self.pred_target == 'x1':
                target = x1_modeled
            elif self.pred_target == 'velocity':
                target = dxt
            elif self.pred_target in ['epsilon', 'eps']:
                target = x0
            else:
                raise ValueError(f"Unknown pred_target {self.pred_target}. Supported: 'x1','velocity','epsilon'/'eps'.")

            # per-sample MSE scalar (mean over all dims except batch)
            per_sample_mse = (model_pred - target).pow(2).mean(dim=tuple(range(1, model_pred.ndim)))

            # --- scheduler outputs (shapes [bs]) ---
            sched_out = self.path.scheduler(t_fm)  # expected to return tensors alpha_t, d_alpha_t, sigma_t, d_sigma_t
            alpha_t = sched_out.alpha_t
            sigma_t = sched_out.sigma_t
            d_alpha_t = sched_out.d_alpha_t
            d_sigma_t = sched_out.d_sigma_t

            # compute lambda' (time derivative of log-SNR)
            # lambda_t = log(alpha_t^2 / sigma_t^2) -> lambda'_t = 2*(d_alpha/alpha - d_sigma/sigma)
            lambda_dot = 2.0 * (d_alpha_t / alpha_t - d_sigma_t / sigma_t)  # shape [bs]

            # numeric safety
            eps_small = 1e-8
            # --- construct desired per-sample weight mult(t) ---
            if self.pred_target in ['epsilon', 'eps']:
                # Theory-backed LCFM weight for epsilon-pred in affine parametrization:
                # w_des(t) ∝ -0.5 * lambda_dot * sigma_t^2
                w_des = (-0.5 * lambda_dot * (sigma_t ** 2))
                # avoid negative or zero weights; clamp to small positive
                w_des = w_des.clamp(min=eps_small)

                # Because we sample uniformly in t, we can directly use w_des as multiplier
                mult = w_des

            elif self.pred_target == 'velocity':
                # Practical, robust option for velocity-pred:
                # 1) estimate per-sample target variance (per-batch) and normalize so loss contributions across t are comparable
                # 2) optionally multiply by an |lambda_dot|-dependent factor to favor SNR-related weighting
                # Estimate per-sample variance (mean squared norm) across feature dims:
                target_var = target.pow(2).mean(dim=tuple(range(1, target.ndim))).clamp(min=eps_small)  # shape [bs]
                # choose a lambda-based factor (abs to avoid sign issues)
                lambda_factor = lambda_dot.abs().clamp(min=eps_small)

                # Compose multiplier: inverse-variance * lambda_factor (and normalize by mean to keep global loss scale stable)
                mult_raw = (1.0 / target_var) * lambda_factor
                mult = mult_raw / mult_raw.mean().clamp(min=eps_small)

            elif self.pred_target == 'x1':
                # For direct x0/x1 prediction (target-prediction), targets often shrink with t (alpha_t),
                # so a robust approach is to normalize by the expected target-energy at this t.
                # If we assume data x0 has variance ~ Var_x (unknown), a simple per-batch estimate is used.
                target_var = target.pow(2).mean(dim=tuple(range(1, target.ndim))).clamp(min=eps_small)  # shape [bs]
                # Optionally include alpha_t factor to reflect signal scaling:
                alpha_factor = alpha_t.abs().clamp(min=eps_small)
                mult_raw = (1.0 / target_var) * alpha_factor
                mult = mult_raw / mult_raw.mean().clamp(min=eps_small)

            else:
                raise ValueError("Unhandled pred_target")

            # finally combine: weighted per-sample MSE averaged to scalar loss
            loss_fm = (per_sample_mse * mult).mean()
            fm_mult_mean = mult.mean().detach()
            fm_lambda_dot_mean = lambda_dot.mean().detach()
        else:

            if self.pred_target == 'x1':
                loss_fm = th.nn.functional.mse_loss(model_pred, x1_modeled)
            elif self.pred_target == 'velocity':
                loss_fm = th.nn.functional.mse_loss(model_pred, dxt)
            elif self.pred_target in ['epsilon', 'eps']:
                # For epsilon prediction, the ground truth is x0 (the noise)
                loss_fm = th.nn.functional.mse_loss(model_pred, x0)
            else:
                raise ValueError(f"Unknown pred_target {self.pred_target}. Supported: 'x1', 'velocity', 'epsilon'/'eps'.")

        # Calculate the predicted x1.
        if self.pred_target == 'x1':
            pred_x1 = model_pred
        elif self.pred_target == 'velocity':

            # Copied from https://github.com/facebookresearch/flow_matching/blob/25ae2d6a672468b58775f47ea086a2a8836be5a4/flow_matching/path/affine.py#L146
            # Cannot directly use self.path.target_to_velocity since it does not handle shape expansion robustly.
            scheduler_output = self.path.scheduler(t_fm)

            alpha_t = scheduler_output.alpha_t
            d_alpha_t = scheduler_output.d_alpha_t
            sigma_t = scheduler_output.sigma_t
            d_sigma_t = scheduler_output.d_sigma_t

            # Calculate coefficients a_t and b_t (these have shape [bs])
            denominator = (d_alpha_t * sigma_t - d_sigma_t * alpha_t)
            a_t = -d_sigma_t / denominator
            b_t = sigma_t / denominator

            # Expand coefficients to match the dimensions of x_t_pose and velocity for broadcasting
            a_t = expand_tensor_like(a_t, xt)
            b_t = expand_tensor_like(b_t, model_pred)

            pred_x1 = a_t * xt + b_t * model_pred
        elif self.pred_target in ['epsilon', 'eps']:
            # Convert epsilon prediction to x1 using epsilon_to_target formula
            # From affine.py: epsilon_to_target: a_t = 1 / alpha_t, b_t = -sigma_t / alpha_t
            scheduler_output = self.path.scheduler(t_fm)

            alpha_t = scheduler_output.alpha_t
            sigma_t = scheduler_output.sigma_t

            # Calculate coefficients a_t and b_t (these have shape [bs])
            a_t = 1.0 / alpha_t
            b_t = -sigma_t / alpha_t

            # Expand coefficients to match the dimensions of x_t and epsilon for broadcasting
            a_t = expand_tensor_like(a_t, xt)
            b_t = expand_tensor_like(b_t, model_pred)

            pred_x1 = a_t * xt + b_t * model_pred
        else:
            raise ValueError(f"Unknown pred_target {self.pred_target}. Supported: 'x1', 'velocity', 'epsilon'/'eps'.")

        enable_smpl_loss = kwargs.get("enable_smpl_loss", False)
        current_step = int(kwargs.get("current_step", 0))
        geom_global_weight, geom_time_weight, geom_level_weights = self._geometry_weights(t_fm, current_step)
        geom_time_weight_mean = geom_time_weight.mean()

        pred_x1_for_geom = pred_x1
        if self.geometry_pred_x1_grad_scale < 1.0:
            pred_x1_for_geom = pred_x1.detach() + self.geometry_pred_x1_grad_scale * (pred_x1 - pred_x1.detach())

        if enable_smpl_loss:
            active_geom_mask = geom_time_weight > 0.0
            if geom_global_weight > 0.0 and active_geom_mask.any():
                batch_geom = self._slice_batch(batch, active_geom_mask)
                with th.amp.autocast(self.device.type, enabled=False):
                    if self.task == 'pose':
                        traj_feats_geom = get_traj_features_from_batch(batch_geom, self.ds, self.task).float()
                    else:
                        traj_feats_geom = None
                    pred_geom = pred_x1_for_geom[active_geom_mask].float()
                    model_output_full = _combine_data(
                        modeled_part=pred_geom,
                        traj_feats=traj_feats_geom,
                        task=self.task,
                        ds=self.ds,
                        predicted_modalities=self.predicted_modalities,
                    )
                    loss_dict = _unwrap_model.compute_losses_with_smpl(batch_geom, model_output_full, smplx_model, epoch=epoch, **kwargs)
                loss_dict = self._scale_loss_dict(loss_dict, _unwrap_model, geom_level_weights, geom_time_weight_mean)
            else:
                loss_dict = {'loss': loss_fm.new_zeros(())}
        else:
            loss_dict = {'loss': loss_fm.new_zeros(())}

        moe_loss_dict = {}
        if hasattr(_unwrap_model, 'get_moe_loss_dict'):
            moe_loss_dict = _unwrap_model.get_moe_loss_dict()
            assert_action_blind_mapping(moe_loss_dict, context="MoE loss dict")
            for key in moe_loss_dict:
                if key.endswith('_per_action') or '_per_action' in key:
                    raise ValueError(f"Forbidden per-action MoE loss key '{key}'.")
            loss_dict.update(moe_loss_dict)

        # NOTE: Add flow/geometry monitor scalars to loss_dict.
        if not self.discrete_timestep_sampling and self.time_sampler == 'uniform':
            loss_dict['fm_mult_mean'] = fm_mult_mean
            loss_dict['fm_lambda_dot_mean'] = fm_lambda_dot_mean
        loss_dict['fm_t_mean'] = t_fm.mean().detach()
        loss_dict['fm_t_min'] = t_fm.min().detach()
        loss_dict['fm_t_max'] = t_fm.max().detach()
        loss_dict['geom_global_weight'] = loss_fm.new_tensor(geom_global_weight)
        loss_dict['geom_time_weight_mean'] = geom_time_weight_mean.detach()
        loss_dict['geom_level1_weight'] = loss_fm.new_tensor(geom_level_weights['level1'])
        loss_dict['geom_level2_weight'] = loss_fm.new_tensor(geom_level_weights['level2'])
        loss_dict['geom_level3_weight'] = loss_fm.new_tensor(geom_level_weights['level3'])

        loss_geom = loss_dict['loss']
        moe_loss_total = sum_loss_dict(moe_loss_dict) if moe_loss_dict else loss_fm.new_zeros(())
        include_moe_in_main = bool(moe_loss_dict) and bool(
            _unwrap_model.moe_aux_loss_in_main_loss() if hasattr(_unwrap_model, 'moe_aux_loss_in_main_loss') else False
        )
        loss_dict['loss_fm_velocity'] = loss_fm
        loss_dict['loss_geom_manifold'] = loss_geom
        if moe_loss_dict:
            loss_dict['moe_aux_total'] = moe_loss_total.detach()
        loss_dict['loss'] = loss_fm + loss_geom + (moe_loss_total if include_moe_in_main else loss_fm.new_zeros(()))

        if kwargs.get("store_pred", False):
            if self.task == 'pose':
                traj_feats = get_traj_features_from_batch(batch, self.ds, self.task)
                batch['extras/pred'] = _combine_data(
                    modeled_part=pred_x1.detach(),
                    traj_feats=traj_feats,
                    task=self.task,
                    ds=self.ds,
                    predicted_modalities=self.predicted_modalities,
                )
            else:
                batch['extras/pred'] = pred_x1.detach()

        if self.log_gradient_norms:
            grad_norm_fm = self._tensor_grad_norm(loss_fm, model_pred).detach()
            grad_norm_geom = self._tensor_grad_norm(loss_geom, model_pred).detach()
            loss_dict['grad_norm_fm_model_pred'] = grad_norm_fm
            loss_dict['grad_norm_geom_model_pred'] = grad_norm_geom
            loss_dict['grad_norm_geom_to_fm'] = grad_norm_geom / grad_norm_fm.clamp(min=1e-12)

        return loss_dict

    @th.no_grad()
    def p_sample_loop(self, model, batch, shape, progress=False, **kwargs):
        """
        Performs sampling for PoseNet using an ODE solver.
        """
        bs, device = shape[0], next(model.parameters()).device
        
        collect_sampling_progress = kwargs.pop('collect_sampling_progress', False)
        sampling_progress_stride = max(1, int(kwargs.pop('sampling_progress_stride', 1)))
        sampling_progress_max_steps = int(kwargs.pop('sampling_progress_max_steps', 0))
        sampling_progress_payload = None

        target_modeled = None
        if collect_sampling_progress:
            motion_repr_clean = batch.get('motion_repr_clean', None)
            if motion_repr_clean is not None and hasattr(self.ds, 'cond_feat_dim'):
                _, target_modeled = _split_data(
                    motion_repr=motion_repr_clean,
                    task=self.task,
                    ds=self.ds,
                    predicted_modalities=self.predicted_modalities,
                )

        def _build_sampling_payload(all_states: th.Tensor, time_points: th.Tensor):
            nonlocal sampling_progress_payload
            if all_states is None:
                return

            total_steps = all_states.shape[0]
            device_local = all_states.device

            candidate_idx = th.arange(total_steps, device=device_local)
            candidate_idx = candidate_idx[::sampling_progress_stride]
            if sampling_progress_max_steps > 0 and candidate_idx.numel() > sampling_progress_max_steps:
                linspace_positions = th.linspace(
                    0,
                    candidate_idx.numel() - 1,
                    steps=sampling_progress_max_steps,
                    device=device_local,
                )
                linspace_positions = linspace_positions.round().long()
                candidate_idx = candidate_idx.index_select(0, linspace_positions)

            states_sel = all_states.index_select(0, candidate_idx).detach()
            times_sel = time_points.index_select(0, candidate_idx).detach()

            rmse = None
            if target_modeled is not None:
                diff = states_sel - target_modeled.unsqueeze(0)
                reduce_dims = tuple(range(2, diff.ndim))
                rmse = diff.pow(2).mean(dim=reduce_dims).sqrt().detach()

            sampling_progress_payload = {
                'modeled_states': states_sel,
                'time_grid': times_sel,
                'rmse_to_target': rmse,
                'sample_indices': candidate_idx.detach(),
            }

        x_modeled = th.randn(*shape, device=device) * self.sampling_noise_scale

        class VelocityModelWrapper(ModelWrapper):
            def __init__(self, model_to_wrap, scheduler=None, pred_target='x1'):
                super().__init__(model_to_wrap)
                self.scheduler = scheduler
                self.pred_target = pred_target

            def forward(self, x: th.Tensor, t: th.Tensor, batch, **extras) -> th.Tensor:

                batch['x_t'] = x
                bs, *_ = batch['x_t'].shape
                t = t[None].expand(x.shape[0])
                model_pred = self.model(batch, t, **extras)

                if self.pred_target == 'x1':
                    # NOTE: copied from https://github.com/facebookresearch/flow_matching/blob/25ae2d6a672468b58775f47ea086a2a8836be5a4/flow_matching/path/affine.py#L94
                    # Since the scheduler output tensors have shape [bs], we need to expand them to match x's shape for broadcasting.
                    scheduler_output = self.scheduler(t)

                    alpha_t = scheduler_output.alpha_t
                    d_alpha_t = scheduler_output.d_alpha_t
                    sigma_t = scheduler_output.sigma_t
                    d_sigma_t = scheduler_output.d_sigma_t

                    a_t = d_sigma_t / sigma_t
                    b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t

                    a_t = expand_tensor_like(a_t, x)
                    b_t = expand_tensor_like(b_t, x)

                    pred_dxt = a_t * x + b_t * model_pred
                elif self.pred_target == 'velocity':
                    pred_dxt = model_pred
                elif self.pred_target in ['epsilon', 'eps']:
                    # Convert epsilon prediction to velocity using epsilon_to_velocity formula
                    # From affine.py: epsilon_to_velocity: a_t = d_alpha_t / alpha_t, b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t
                    scheduler_output = self.scheduler(t)

                    alpha_t = scheduler_output.alpha_t
                    d_alpha_t = scheduler_output.d_alpha_t
                    sigma_t = scheduler_output.sigma_t
                    d_sigma_t = scheduler_output.d_sigma_t

                    a_t = d_alpha_t / alpha_t
                    b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t

                    a_t = expand_tensor_like(a_t, x)
                    b_t = expand_tensor_like(b_t, model_pred)

                    pred_dxt = a_t * x + b_t * model_pred
                else:
                    raise ValueError(f"Unknown pred_target {self.pred_target}. Supported: 'x1', 'velocity', 'epsilon'/'eps'.")

                return pred_dxt

        score_guided_sampling = kwargs.get('score_guided_sampling', False)
        if score_guided_sampling:
            smplx_model = kwargs.get('smplx_model', None)
            assert smplx_model is not None, "When using score_guided_sampling, smplx_model must be provided."

        if self.ode_solver is None:
            velocity_model = VelocityModelWrapper(model, scheduler=self.path.scheduler, pred_target=self.pred_target)
            self.ode_solver = ODESolver(velocity_model)

        if self.ode_solver_name == 'discrete_euler':
            # Use Discrete Euler solver for discrete timestep sampling. Test-time
            # num_inference_steps can intentionally differ from the training grid.
            num_inference_steps = int(kwargs.get('num_inference_steps', self.num_total_steps))
            step_size = 1.0 / num_inference_steps

            recorded_states = [] if collect_sampling_progress else None
            recorded_times = [] if collect_sampling_progress else None
            if collect_sampling_progress:
                recorded_states.append(x_modeled.clone())
                recorded_times.append(th.tensor(0.0, device=device))
            
            for i in range(num_inference_steps):
                t = th.full((bs,), i * step_size, device=device)
                batch_sample = batch.copy()
                batch_sample['x_t'] = x_modeled
                s = model(batch_sample, t)

                with th.enable_grad():
                    if score_guided_sampling:
                        x_modeled = x_modeled.detach().requires_grad_()
                    # Calculate the predicted x1.
                    if self.pred_target == 'x1':
                        # target to velocity.
                        scheduler_output = self.path.scheduler(t)

                        alpha_t = scheduler_output.alpha_t
                        d_alpha_t = scheduler_output.d_alpha_t
                        sigma_t = scheduler_output.sigma_t
                        d_sigma_t = scheduler_output.d_sigma_t

                        a_t = d_sigma_t / sigma_t
                        b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t

                        a_t = expand_tensor_like(a_t, x_modeled)
                        b_t = expand_tensor_like(b_t, s)
                        pred_dxt = a_t * x_modeled + b_t * s
                        pred_x1 = s
                    elif self.pred_target == 'velocity':
                        # velocity to target.
                        scheduler_output = self.path.scheduler(t)

                        alpha_t = scheduler_output.alpha_t
                        d_alpha_t = scheduler_output.d_alpha_t
                        sigma_t = scheduler_output.sigma_t
                        d_sigma_t = scheduler_output.d_sigma_t

                        # Calculate coefficients a_t and b_t (these have shape [bs])
                        denominator = (d_alpha_t * sigma_t - d_sigma_t * alpha_t)
                        a_t = -d_sigma_t / denominator
                        b_t = sigma_t / denominator

                        # Expand coefficients to match the dimensions of x_t_pose and velocity for broadcasting
                        a_t = expand_tensor_like(a_t, x_modeled)
                        b_t = expand_tensor_like(b_t, s)
                        pred_x1 = a_t * x_modeled + b_t * s
                        pred_dxt = s
                    elif self.pred_target in ['epsilon', 'eps']:
                        # epsilon to velocity and epsilon to target.
                        scheduler_output = self.path.scheduler(t)

                        alpha_t = scheduler_output.alpha_t
                        d_alpha_t = scheduler_output.d_alpha_t
                        sigma_t = scheduler_output.sigma_t
                        d_sigma_t = scheduler_output.d_sigma_t

                        # Convert epsilon to velocity for ODE step
                        # From affine.py: epsilon_to_velocity: a_t = d_alpha_t / alpha_t, b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t
                        a_t_vel = d_alpha_t / alpha_t
                        b_t_vel = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t
                        
                        a_t_vel = expand_tensor_like(a_t_vel, x_modeled)
                        b_t_vel = expand_tensor_like(b_t_vel, s)
                        pred_dxt = a_t_vel * x_modeled + b_t_vel * s

                        # Convert epsilon to target for score-guided sampling
                        # From affine.py: epsilon_to_target: a_t = 1 / alpha_t, b_t = -sigma_t / alpha_t
                        a_t_target = 1.0 / alpha_t
                        b_t_target = -sigma_t / alpha_t
                        
                        a_t_target = expand_tensor_like(a_t_target, x_modeled)
                        b_t_target = expand_tensor_like(b_t_target, s)
                        pred_x1 = a_t_target * x_modeled + b_t_target * s
                    else:
                        raise ValueError(f"Unknown pred_target {self.pred_target}. Supported: 'x1', 'velocity', 'epsilon'/'eps'.")

                    if score_guided_sampling and i > num_inference_steps - 15:
                        traj_feats_guided = get_traj_features_from_batch(batch, self.ds, self.task) if self.task == 'pose' else None
                        model_output_full = _combine_data(modeled_part=pred_x1, traj_feats=traj_feats_guided, task=self.task, ds=self.ds, predicted_modalities=self.predicted_modalities)
                        score = model.guide_skating_with_smpl(batch=batch_sample, model_output=model_output_full, inputs=x_modeled, smplx_model=smplx_model, compute_grad='x_t')
                        # breakpoint()
                        pred_dxt = pred_dxt + 0 * score  # guidance scale = 0.1

                x_modeled = x_modeled + step_size * pred_dxt
                if collect_sampling_progress:
                    recorded_states.append(x_modeled.clone())
                    recorded_times.append(th.tensor((i + 1) * step_size, device=device))
            pred_x1_modeled = x_modeled

            if collect_sampling_progress:
                all_states = th.stack(recorded_states, dim=0)
                time_grid = th.stack(recorded_times, dim=0)
                _build_sampling_payload(all_states, time_grid)
        else:
            # Sample using the ODE solver
            # NOTE: it would be better to use time_grid of [eps_t, 1.0 - eps_t] to avoid numerical instability. 
            
            # Determine step_size and time_grid based on CLI arguments
            step_size = self.ode_step_size if self.ode_step_size is not None else None
            
            if self.ode_time_grid_steps is not None:
                # Create uniform time_grid with specified number of steps
                time_grid = th.linspace(self.eps_t, 1.0 - self.eps_t, self.ode_time_grid_steps, device=device)
            else:
                # Default: use endpoints only (original behavior)
                time_grid = th.tensor([self.eps_t, 1.0 - self.eps_t], device=device)
            
            pred_x1_modeled = self.ode_solver.sample(
                x_modeled,
                step_size=step_size,
                method=self.ode_solver_name,
                time_grid=time_grid,
                return_intermediates=collect_sampling_progress,
                batch=batch,
            )
            if collect_sampling_progress:
                _build_sampling_payload(pred_x1_modeled, time_grid)
                pred_x1_modeled = pred_x1_modeled[-1]

        traj_feats_final = get_traj_features_from_batch(batch, self.ds, self.task) if self.task == 'pose' else None
        combined_output = _combine_data(modeled_part=pred_x1_modeled, traj_feats=traj_feats_final, task=self.task, ds=self.ds, predicted_modalities=self.predicted_modalities)

        if sampling_progress_payload is not None:
            return combined_output, sampling_progress_payload
        return combined_output

    def eval_losses(self, model, batch, shape, smplx_model, compute_eval_metrics=False, epoch=0, **kwargs):
        _unwrap_model = model.module if hasattr(model, 'module') else model
        model_output_full = self.p_sample_loop(model=model, batch=batch, shape=shape, **kwargs)
        loss_dict = _unwrap_model.compute_losses_with_smpl(batch, model_output_full, smplx_model, epoch=epoch, compute_eval_metrics=compute_eval_metrics, **kwargs)
        return loss_dict, model_output_full
