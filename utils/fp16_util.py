# diffusion/fp16_util.py
import torch
from torch.cuda.amp import GradScaler

class MixedPrecisionTrainer:
    """
    A helper class to manage mixed-precision training with gradient scaling and clipping.
    This class encapsulates the standard PyTorch AMP workflow.

    If use_fp16 is False, this class acts as a pass-through for standard FP32 training.
    """
    def __init__(self, model, use_fp16=False, fp16_scale_growth=1e-3, amp_dtype: torch.dtype | None = torch.float16, grad_norm:bool = False, init_loss_weights: dict[str, float] | None = None):
        self.model = model
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.amp_dtype = amp_dtype if self.use_fp16 else None

        self.master_params = list(model.parameters())
        if self.use_fp16 and self.amp_dtype == torch.float16:
            self.scaler = GradScaler(growth_factor=fp16_scale_growth)
            print("MixedPrecisionTrainer: AMP enabled with FP16 + GradScaler.")
        elif self.use_fp16 and self.amp_dtype == torch.bfloat16:
            self.scaler = None
            print("MixedPrecisionTrainer: AMP enabled with BF16 (no GradScaler).")
        else:
            self.scaler = None
            print("MixedPrecisionTrainer: Running in standard FP32 mode.")
        
        self.grad_norm = grad_norm
        if self.grad_norm:
            from utils.grad_norm import GradNormLossWeighter
            # Initialize GradNormLossWeighter with provided configuration
            gn_kwargs = {}

            loss_names = tuple(init_loss_weights.keys())
            weights = list(init_loss_weights.values())
            gn_num_losses = len(weights)
            gn_kwargs.update({
                'loss_names': loss_names,
                'loss_weights': weights,
                'num_losses': gn_num_losses
            })

            self.grad_norm_weighter = GradNormLossWeighter(
                grad_norm_parameters=None,
                **gn_kwargs
            )

    def zero_grad(self):
        """ Zeros the gradients of the model parameters. """
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def backward(self, loss, activations: torch.Tensor | None = None, freeze: bool = False, scale: float = 1.0, grad_step: bool = True, **backward_kwargs):
        """
        Backward pass.

        Behavior:
        - If self.grad_norm is True:
            - `loss` should be a dict[str, Tensor] (or any input accepted by GradNormLossWeighter).
            - Calls GradNormLossWeighter.forward(...) which handles both the weighted loss backward and
              the grad-norm loss backward that updates the hyper-parameters (loss weights).
            - In FP16 mode, wrap the GradNormLossWeighter's `accelerator` so that its internal backward
              calls are performed with self.scaler.scale(...).backward(...).
        - If self.grad_norm is False:
            - `loss` should be a Tensor and is backward-ed normally (scaled if FP16).
        """
        # GradNorm flow
        if self.grad_norm:
            # Basic type check (be permissive; GradNorm accepts lists/tuples/tensors too)
            if not isinstance(loss, (dict, list, tuple, torch.Tensor)):
                raise TypeError("When grad_norm=True, `loss` must be dict[str,Tensor] or list/tuple/Tensor compatible with GradNormLossWeighter")

            # call gradnorm forward which internally performs the necessary backward passes
            # pass activations/freeze/scale/grad_step through as appropriate
            return self.grad_norm_weighter.forward(
                losses=loss,
                activations=activations,
                freeze=freeze,
                scale=scale,
                grad_step=grad_step,
                **backward_kwargs
            )

        # Standard single-loss flow
        else:
            if not isinstance(loss, torch.Tensor):
                raise TypeError("When grad_norm=False, `loss` must be a torch.Tensor")

            if self.scaler is not None:
                self.scaler.scale(loss).backward(**backward_kwargs)
            else:
                loss.backward(**backward_kwargs)

    def optimize(self, opt: torch.optim.Optimizer, max_norm: float):
        """
        Performs the optimization step, including gradient unscaling and clipping
        in the correct order.

        Args:
            opt: The PyTorch optimizer.
            max_norm: The maximum norm for gradient clipping. If 0 or less, no clipping is performed.
        """
        if self.scaler is not None:
            # 1. Unscale gradients computed by scaler.scale(loss).backward()
            self.scaler.unscale_(opt)
            
            # 2. Clip gradients of the model parameters (which are now unscaled)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            # 3. Optimizer step. The scaler skips the step if grads contain infs/NaNs.
            self.scaler.step(opt)
            
            # 4. Update the scale for next iteration.
            self.scaler.update()
        else:  # Standard FP32 training
            # 1. Clip gradients directly
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            # 2. Optimizer step
            opt.step()
