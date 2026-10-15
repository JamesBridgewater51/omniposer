# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

"""
Model factory utilities.

This module provides two interfaces for model creation:
1. OmegaConf-based (for training with Hydra): Uses construct_class_by_name pattern
2. Legacy args-based (for testing): Uses create_model factory function

"""

from typing import Any, Dict
from utils.construct import construct_class_by_name


# =============================================================================
# OmegaConf-based Model Creation (for Hydra training)
# =============================================================================

def create_model_from_config(config: Dict[str, Any], **runtime_kwargs) -> Any:
    """
    Create a model from an OmegaConf-style configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'class_name' and constructor kwargs.
        **runtime_kwargs: Runtime parameters (dataset, device, train_dataloader, etc.).
    
    Returns:
        Instantiated model object.
    """
    config = dict(config)
    config.update(runtime_kwargs)
    class_name = config.pop('class_name')
    print(f"[INFO] Instantiating: {class_name}")
    return construct_class_by_name(class_name=class_name, **config)


def prepare_diffusion_config(config: Dict[str, Any], gd_module) -> Dict[str, Any]:
    """
    Prepare diffusion-specific parameters that require runtime computation.
    
    Computes betas, use_timesteps, model_mean_type, model_var_type, loss_type
    from the config's noise_schedule and sigma_small settings.
    
    Args:
        config: Configuration dictionary (modified in-place).
        gd_module: The gaussian_diffusion module.
    
    Returns:
        Modified config dictionary with computed parameters.
    """
    from model.generative.diffusion.respace import space_timesteps
    
    steps = config['num_diffusion_timesteps']
    noise_schedule = config['noise_schedule']
    sigma_small = config['sigma_small']
    
    # Compute betas
    config['betas'] = gd_module.get_named_beta_schedule(noise_schedule, steps, 1.0)
    
    # Compute use_timesteps
    timestep_respacing = config.get('timestep_respacing', '')
    if not timestep_respacing:
        timestep_respacing = [steps]
    config['use_timesteps'] = space_timesteps(steps, timestep_respacing)
    
    # Set enum types
    config['model_mean_type'] = gd_module.ModelMeanType.START_X
    config['model_var_type'] = (
        gd_module.ModelVarType.FIXED_SMALL if sigma_small 
        else gd_module.ModelVarType.FIXED_LARGE
    )
    config['loss_type'] = gd_module.LossType.MSE
    config['rescale_timesteps'] = False
    
    return config


# =============================================================================
# Legacy args-based Model Creation (for testing)
# =============================================================================

def _create_diffusion_model(args, spaced_class, gd_module, **common_kwargs):
    """ Helper function to create a Gaussian Diffusion model. """
    from model.generative.diffusion.respace import space_timesteps
    predict_xstart = True
    steps = common_kwargs["num_diffusion_timesteps"]
    scale_beta = 1.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd_module.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd_module.LossType.MSE
    
    timestep_respacing = common_kwargs.get("timestep_respacing", '')
    if not timestep_respacing:
        timestep_respacing = [steps]

    return spaced_class(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_module.ModelMeanType.EPSILON if not predict_xstart else gd_module.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_module.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd_module.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_module.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        **common_kwargs
    )

def _create_meanflow_model(args, spaced_class, gd_module, **common_kwargs):
    """ Helper function to create a MeanFlow model. """
    # Only pass arguments actually used by SpacedMeanFlowTrajNet/MeanFlowTrajNet
    return spaced_class(
        flow_ratio=args.flow_ratio,
        time_dist_type=args.time_dist_type,
        time_dist_mu=args.time_dist_mu,
        time_dist_sigma=args.time_dist_sigma,
        **common_kwargs
    )

def _create_shortcut_model(args, spaced_class, gd_module, **common_kwargs):
    """ Helper function to create a Shortcut model. """
    # Similar to MeanFlow, this uses a pass-through wrapper for API compatibility.
    return spaced_class(
        # Real Shortcut args
        shortcut_steps=args.shortcut_steps,
        shortcut_empirical_ratio=args.shortcut_empirical_ratio,
        **common_kwargs
    )

def _create_consistency_model(args, spaced_class, gd_module, **common_kwargs):
    """ Helper function to create a Consistency model. """
    # These are for API compatibility with the factory, not used by CM
    # betas, model_mean_type, model_var_type, loss_type, rescale_timesteps
    return spaced_class(
        sigma_min=args.cm_sigma_min,
        sigma_max=args.cm_sigma_max,
        rho=args.cm_rho,
        loss_norm=args.cm_loss_norm,
        betas=None,
        model_mean_type=None,
        model_var_type=None,
        loss_type=None,
        rescale_timesteps=False,
        **common_kwargs
    )

def _create_improved_consistency_model(args, spaced_class, gd_module, **common_kwargs):
    """ Helper function to create an Improved Consistency model. """
    return spaced_class(
        sigma_min=args.cm_sigma_min,
        sigma_max=args.cm_sigma_max,
        rho=args.cm_rho,
        loss_norm=args.cm_loss_norm,
        icm_initial_timesteps=args.icm_initial_timesteps,
        icm_final_timesteps=args.icm_final_timesteps,
        icm_lognormal_mean=args.icm_lognormal_mean,
        icm_lognormal_std=args.icm_lognormal_std,
        # Dummy args for API compatibility
        betas=None,
        model_mean_type=None,
        model_var_type=None,
        loss_type=None,
        rescale_timesteps=False,
        **common_kwargs
    )

def _create_flow_matching_model(args, spaced_class, gd_module, **common_kwargs):
    """ Helper function to create a Flow Matching model. """
    # Build scheduler_kwargs dictionary based on scheduler type
    scheduler_kwargs = {}
    if args.fm_scheduler_type == 'polynomial':
        scheduler_kwargs['polynomial_n'] = args.fm_polynomial_n
    elif args.fm_scheduler_type == 'vp':
        scheduler_kwargs['vp_beta_min'] = args.fm_vp_beta_min
        scheduler_kwargs['vp_beta_max'] = args.fm_vp_beta_max
    
    return spaced_class(
        # Real Shortcut args
        num_total_steps=args.fm_total_steps,
        ode_solver=args.fm_ode_solver,
        ode_step_size=args.fm_ode_step_size,
        ode_time_grid_steps=args.fm_ode_time_grid_steps,
        logit_m=args.fm_logit_m,
        logit_s=args.fm_logit_s,
        mode_s=args.fm_mode_s,
        eps_t=args.fm_eps_t,
        time_sampler=args.fm_time_sampler,
        discrete_timestep_sampling=args.fm_use_discrete_timestep_sampling,
        pred_target=args.fm_pred_target,
        scheduler_type=args.fm_scheduler_type,
        scheduler_kwargs=scheduler_kwargs,
        **common_kwargs
    )

# --- Refactoring Note: A registry (dictionary) of model creator functions ---
GENERATIVE_MODEL_CREATORS = {
    'diffusion': _create_diffusion_model,
    'meanflow': _create_meanflow_model,
    'shortcut': _create_shortcut_model,
    'consistency': _create_consistency_model,
    'flow_matching': _create_flow_matching_model,  # Alias for meanflow
    'improved_consistency': _create_improved_consistency_model,
}


def _create_discriminative_toy_model(args, spaced_class, **common_kwargs):
    """ Helper function to create a Toy discriminative model. """
    return spaced_class(
        **common_kwargs
    )
DISCRIMINATIVE_MODEL_CREATORS = {
    'toy': _create_discriminative_toy_model,

}

def create_generative_model(
    args,
    spaced_classes_map,
    gd_diffusion_module,
    **common_kwargs
):
    """
    A factory function to create a generative model (Diffusion, MeanFlow, etc.)
    by dispatching to a specialized helper function.
    """
    model_type = args.model_type
    
    creator_fn = GENERATIVE_MODEL_CREATORS.get(model_type)
    if creator_fn is None:
        # The original code had an empty elif for 'consistency', which would fail later.
        # This makes the failure explicit and informative.
        raise ValueError(f"Unknown generative model type: {model_type}")

    spaced_class = spaced_classes_map.get(model_type)
    if spaced_class is None:
        raise ValueError(f"No spaced class provided for model type: {model_type}")

    # Pass the gd_diffusion_module (e.g., gaussian_diffusion_trajnet) which is needed by all helpers
    # for utility functions like get_named_beta_schedule.
    return creator_fn(args, spaced_class, gd_diffusion_module, **common_kwargs)


def create_discriminative_model(
    args,
    spaced_classes_map,
    **common_kwargs
):
    """
    A factory function to create a generative model (Diffusion, MeanFlow, etc.)
    by dispatching to a specialized helper function.
    """
    model_type = args.model_type

    creator_fn = DISCRIMINATIVE_MODEL_CREATORS.get(model_type)
    if creator_fn is None:
        # The original code had an empty elif for 'consistency', which would fail later.
        # This makes the failure explicit and informative.
        raise ValueError(f"Unknown discriminative model type: {model_type}")
    spaced_class = spaced_classes_map.get(model_type)
    if spaced_class is None:
        raise ValueError(f"No spaced class provided for model type: {model_type}")

    # Pass the gd_diffusion_module (e.g., gaussian_diffusion_trajnet) which is needed by all helpers
    # for utility functions like get_named_beta_schedule.
    return creator_fn(args, spaced_class, **common_kwargs)


def create_model(args, spaced_classes_map, gd_diffusion_module, **common_kwargs):
    """
    Factory function to create a model (either generative or discriminative).
    
    Dispatches to create_generative_model or create_discriminative_model based on
    args.abstract_model_type.
    
    Args:
        args: Configuration object with model parameters.
        spaced_classes_map: Dict mapping model_type to spaced class.
        gd_diffusion_module: The gaussian_diffusion module (for generative models).
        **common_kwargs: Additional kwargs passed to the model constructor.
    
    Returns:
        Instantiated model object.
    """
    if args.abstract_model_type == "generative":
        return create_generative_model(args, spaced_classes_map, gd_diffusion_module, **common_kwargs)
    elif args.abstract_model_type == "discriminative":
        return create_discriminative_model(args, spaced_classes_map, **common_kwargs)
    else:
        raise ValueError(f"Unknown abstract_model_type: {args.abstract_model_type}")
