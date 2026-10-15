import torch
from tqdm import tqdm

from train.training_loop import TrainLoop
from utils import dist_util


class TrainLoopMaskPerRank(TrainLoop):
    """
    Specialized training loop: one masking schedule per DDP rank.
    Each rank runs exactly one masking action per step; DDP all-reduce
    averages gradients across masking schedules automatically.
    """

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        if config is not None:
            self._ensure_masking_defaults(config)
        super().__init__(*args, **kwargs)
        self._init_mask_per_rank()

    def _ensure_masking_defaults(self, config):
        from omegaconf import open_dict

        masking = getattr(config, "masking", None)
        if masking is None:
            return
        actions = masking.get("actions") if hasattr(masking, "get") else None
        if not actions:
            return
        with open_dict(masking):
            masking["same_batch_for_schedules"] = True
            masking["schedule_per_step"] = "all"
            if "mixture" not in masking:
                masking["mixture"] = [{
                    "until_step": None,
                    "weights": {action["name"]: 1.0 for action in actions},
                }]

    def _init_mask_per_rank(self):
        if not self.use_ddp:
            raise ValueError("TrainLoopMaskPerRank requires DDP (ddp.enabled=true).")

        actions = getattr(self.config, "masking", {}).get("actions", None)
        if not actions:
            raise ValueError("masking.actions must be defined for mask_per_rank training.")

        self._mask_action_names = [action["name"] for action in actions]
        if not self._mask_action_names:
            raise ValueError("masking.actions is empty; cannot assign masks to ranks.")
        if self.world_size != len(self._mask_action_names):
            raise ValueError(
                f"world_size ({self.world_size}) must match num masking actions "
                f"({len(self._mask_action_names)}) for mask_per_rank training."
            )

    def _masking_action_for_rank(self):
        action_name = self._mask_action_names[self.rank]
        return self._materialize_mask_action(action_name)

    def run_loop(self):
        start_epoch = 0
        batches_per_opt_step = 1
        if self.batches_per_epoch_per_gpu > 0 and batches_per_opt_step > 0:
            num_optimizer_steps_per_epoch = self.batches_per_epoch_per_gpu // batches_per_opt_step
            if num_optimizer_steps_per_epoch > 0:
                start_epoch = int(self.step // num_optimizer_steps_per_epoch)
        else:
            num_optimizer_steps_per_epoch = 0

        for epoch in range(start_epoch, self.num_epochs):
            if self.task == 'both':
                self.trajnet.train()
                self.posenet.train()
            else:
                self.model.train()
            if self.use_ddp:
                self.train_dl.sampler.set_epoch(epoch)

            data_iterator = iter(self.train_dl)
            pbar = tqdm(range(num_optimizer_steps_per_epoch), disable=self.disable_tqdm)

            for _ in pbar:
                if self.step >= self.num_steps:
                    return

                if self.step % self.eval_interval == 0:
                    if self.task == 'both':
                        self.evaluate(epoch, save_best_model=False)
                        pipeline_metric = self._evaluate_pipeline(epoch)
                        current_stage = self.get_current_stage(self.step)
                        if not hasattr(self, 'best_pipeline_loss_per_stage'):
                            self.best_pipeline_loss_per_stage = {stage: float('inf') for stage in range(self.masking_num_stages)}

                        if pipeline_metric.item() < self.best_pipeline_loss_per_stage.get(current_stage, float('inf')):
                            self.best_pipeline_loss_per_stage[current_stage] = pipeline_metric.item()
                            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                                self.logger.info(
                                    f"[*] New best model for STAGE {current_stage} found with "
                                    f"Pipeline MPJPE {pipeline_metric.item():.4f} mm. Saving checkpoint."
                                )
                            self.save(is_best_for_stage=True, stage=current_stage)
                    else:
                        self.evaluate(epoch)

                masking_action = self._masking_action_for_rank()

                if self.task == 'both':
                    self.traj_mp_trainer.zero_grad()
                    self.pose_mp_trainer.zero_grad()
                else:
                    self.mp_trainer.zero_grad()

                accumulated_losses = {}
                accumulated_counts = {}
                log_this_step = self.step % self.log_interval == 0
                compute_metrics = log_this_step

                def _accumulate(prefix, losses):
                    for key, val in losses.items():
                        full_key = f"{prefix}{key}"
                        accumulated_losses[full_key] = accumulated_losses.get(full_key, 0.0) + val
                        accumulated_counts[full_key] = accumulated_counts.get(full_key, 0) + 1

                try:
                    batch = next(data_iterator)
                except StopIteration:
                    break

                if self.task == 'both':
                    if self.cascade:
                        losses_log = self.run_micro_batch(
                            batch,
                            masking_action,
                            epoch,
                            task='both',
                            compute_metrics=compute_metrics,
                        )
                        _accumulate('both/', losses_log)
                    else:
                        traj_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        pose_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        traj_losses_log = self.run_micro_batch(
                            traj_batch,
                            masking_action,
                            epoch,
                            task='traj',
                            compute_metrics=compute_metrics,
                        )
                        pose_losses_log = self.run_micro_batch(
                            pose_batch,
                            masking_action,
                            epoch,
                            task='pose',
                            compute_metrics=compute_metrics,
                        )
                        _accumulate('traj/', traj_losses_log)
                        _accumulate('pose/', pose_losses_log)
                else:
                    losses_log = self.run_micro_batch(
                        batch,
                        masking_action,
                        epoch,
                        task=self.task,
                        compute_metrics=compute_metrics,
                    )
                    _accumulate(f'{self.task}/', losses_log)

                if self.task == 'both':
                    self.traj_mp_trainer.optimize(self.traj_opt, max_norm=self.grad_clip_val)
                    self.pose_mp_trainer.optimize(self.pose_opt, max_norm=self.grad_clip_val)
                else:
                    self.mp_trainer.optimize(self.opt, max_norm=self.grad_clip_val)

                self._update_ema()
                if self.model_type == 'consistency':
                    self._update_target_ema()

                if self.task == 'both':
                    self.traj_scheduler.step()
                    self.pose_scheduler.step()
                else:
                    self.scheduler.step()

                if self.step % self.log_interval == 0:
                    avg_losses = {
                        k: v / max(1, accumulated_counts.get(k, 1))
                        for k, v in accumulated_losses.items()
                    }
                    if not self.use_ddp or (self.use_ddp and self.rank == 0):
                        for key, val in avg_losses.items():
                            self.writer.add_scalar(f'train/{key}', val, self.step)
                            self.logger.info(f'[Step {self.step}/{self.num_steps}] [train] {key}: {val:.10f}')

                        if self.task == 'both':
                            self.writer.add_scalar('train/traj_lr', self.traj_opt.param_groups[0]['lr'], self.step)
                            self.writer.add_scalar('train/pose_lr', self.pose_opt.param_groups[0]['lr'], self.step)
                        else:
                            self.writer.add_scalar('train/lr', self.opt.param_groups[0]['lr'], self.step)

                if self.step > 0 and self.step % self.save_interval == 0:
                    self.save()
                self.step += 1

        if self.use_ddp:
            dist_util.cleanup()
