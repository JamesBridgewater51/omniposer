"""W&B logging adapter with the project's legacy writer-like surface.

The project historically passed file-backed scalar writers through many training
loops. This module keeps that call surface while enforcing W&B's single-run model
and DDP rank-zero safety.
"""

from __future__ import annotations

import html
import os
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in RoHM, but keep imports robust.
    torch = None


_ACTIVE_WRITER: "WandbWriter | None" = None
_OWNS_ACTIVE_RUN = False


class _WandbRunState:
    """Shared W&B run state used by parent and prefixed child writers."""

    def __init__(self, run: Any):
        self.run = run
        self.pending_step: Optional[int] = None
        self.pending_payload: dict[str, Any] = {}

    def log(self, payload: Mapping[str, Any], step: Optional[int] = None, commit: Optional[bool] = None):
        if commit is not None or step is None:
            self.flush()
            kwargs = {}
            if step is not None:
                kwargs["step"] = step
            if commit is not None:
                kwargs["commit"] = commit
            self.run.log(dict(payload), **kwargs)
            return

        # Legacy writer callers frequently emit many add_scalar/add_* calls with
        # the same global_step. Buffer those values into one W&B history row so
        # W&B's monotonic step semantics do not drop same-step metrics.
        if self.pending_step is not None and step != self.pending_step:
            self.flush()
        self.pending_step = step
        self.pending_payload.update(dict(payload))

    def flush(self):
        if self.pending_payload:
            self.run.log(self.pending_payload, step=self.pending_step)
            self.pending_payload = {}
            self.pending_step = None

    def finish(self):
        self.flush()
        self.run.finish()


def _wandb():
    import wandb

    return wandb


def _is_disabled_mode(mode: Optional[str]) -> bool:
    return str(mode or "").lower() == "disabled"


def _to_python(value: Any) -> Any:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.numpy()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)
    return value


def _sanitize_config(config: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except Exception:
        pass
    if hasattr(config, "__dict__") and not isinstance(config, type):
        return vars(config)
    return config


def _sanitize_wandb_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[\\/#?%:]", "-", text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-_.")
    return text[:128]


def _stable_run_identity(name: Optional[str], run_id: Optional[str], logdir: Optional[str]) -> tuple[str, str]:
    base = name or run_id or (os.path.basename(os.path.normpath(logdir)) if logdir else "rohm-run")
    display_name = str(base).strip() or "rohm-run"
    stable_id = _sanitize_wandb_id(run_id or display_name)
    if not stable_id:
        stable_id = datetime.now().strftime("rohm-run-%Y%m%d-%H%M%S")
    return display_name, stable_id


def _normalize_resume(resume: Optional[str]) -> str:
    return str(resume or os.environ.get("WANDB_RESUME", "allow"))


def _to_wandb_image_array(img: Any, dataformats: str = "CHW") -> Any:
    arr = _to_numpy(img)
    fmt = (dataformats or "CHW").upper()

    if arr.ndim == 4:
        return [_to_wandb_image_array(item, fmt[1:] if fmt.startswith("N") else fmt) for item in arr]
    if arr.ndim == 3:
        if fmt == "CHW":
            # Legacy image tensors are commonly CHW; W&B expects HWC.
            arr = np.transpose(arr, (1, 2, 0))
        elif fmt == "HWC":
            pass
        elif fmt == "HW":
            arr = arr.squeeze()
        else:
            # Fall back to channel-first if the first axis looks like C.
            if arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
    return arr


def _to_wandb_video_array(video: Any, dataformats: str = "NTCHW") -> np.ndarray:
    arr = _to_numpy(video)
    fmt = (dataformats or "NTCHW").upper()
    if arr.ndim == 4:
        # Legacy video calls use TCHW for a single video. W&B accepts
        # NTHWC/NTCHW-like arrays, so add a batch dimension and preserve channels.
        if fmt == "TCHW":
            arr = arr[None]
        elif fmt == "THWC":
            arr = np.transpose(arr, (0, 3, 1, 2))[None]
        else:
            arr = arr[None]
    elif arr.ndim == 5 and fmt in {"NTHWC", "TNHWC"}:
        if fmt == "TNHWC":
            arr = np.transpose(arr, (1, 0, 4, 2, 3))
        else:
            arr = np.transpose(arr, (0, 1, 4, 2, 3))
    return arr


def _split_mesh_batch(value: Any) -> list[np.ndarray]:
    arr = _to_numpy(value)
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    return [arr]


class NullWandbWriter:
    """No-op writer for non-main DDP ranks and disabled logging."""

    is_active = False

    def add_scalar(self, *args, **kwargs):
        return None

    def add_scalars(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def add_image(self, *args, **kwargs):
        return None

    def add_images(self, *args, **kwargs):
        return None

    def add_video(self, *args, **kwargs):
        return None

    def add_audio(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def add_mesh(self, *args, **kwargs):
        return None

    def add_hparams(self, *args, **kwargs):
        return None

    def watch(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def with_prefix(self, *args, **kwargs):
        return self

    def flush(self):
        return None

    def close(self):
        return None


class WandbWriter:
    """Legacy writer-compatible wrapper around a single active W&B run."""

    is_active = True

    def __init__(self, run: Any, prefix: str = "", owns_run: bool = False, state: Optional[_WandbRunState] = None):
        self._state = state or _WandbRunState(run)
        self.run = self._state.run
        self.prefix = prefix.strip("/")
        self.owns_run = owns_run

    def _key(self, tag: str) -> str:
        tag = str(tag).strip("/")
        return f"{self.prefix}/{tag}" if self.prefix and tag else self.prefix or tag

    def log(self, data: Mapping[str, Any], step: Optional[int] = None, commit: Optional[bool] = None):
        payload = {self._key(k): _to_python(v) for k, v in data.items()}
        self._state.log(payload, step=step, commit=commit)

    def add_scalar(self, tag: str, scalar_value: Any, global_step: Optional[int] = None, *args, **kwargs):
        self.log({tag: scalar_value}, step=global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: Mapping[str, Any], global_step: Optional[int] = None, *args, **kwargs):
        # Legacy grouped scalar calls map naturally to W&B slash prefixes, so
        # flatten `main_tag/{sub_key}` into one log call.
        self.log({f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}, step=global_step)

    def add_histogram(self, tag: str, values: Any, global_step: Optional[int] = None, *args, **kwargs):
        wandb = _wandb()
        self._state.log({self._key(tag): wandb.Histogram(_to_numpy(values))}, step=global_step)

    def add_image(self, tag: str, img_tensor: Any, global_step: Optional[int] = None, dataformats: str = "CHW", *args, **kwargs):
        wandb = _wandb()
        image = _to_wandb_image_array(img_tensor, dataformats=dataformats)
        if isinstance(image, list):
            value = [wandb.Image(item) for item in image]
        else:
            value = wandb.Image(image)
        self._state.log({self._key(tag): value}, step=global_step)

    def add_images(self, tag: str, img_tensor: Any, global_step: Optional[int] = None, dataformats: str = "NCHW", *args, **kwargs):
        wandb = _wandb()
        images = _to_wandb_image_array(img_tensor, dataformats=dataformats)
        if not isinstance(images, list):
            images = [images]
        self._state.log({self._key(tag): [wandb.Image(image) for image in images]}, step=global_step)

    def add_video(self, tag: str, vid_tensor: Any, global_step: Optional[int] = None, fps: int = 4, dataformats: str = "NTCHW", *args, **kwargs):
        wandb = _wandb()
        video = _to_wandb_video_array(vid_tensor, dataformats=dataformats)
        self._state.log({self._key(tag): wandb.Video(video, fps=fps)}, step=global_step)

    def add_audio(self, tag: str, snd_tensor: Any, global_step: Optional[int] = None, sample_rate: int = 44100, *args, **kwargs):
        wandb = _wandb()
        audio = _to_numpy(snd_tensor).squeeze()
        self._state.log({self._key(tag): wandb.Audio(audio, sample_rate=sample_rate)}, step=global_step)

    def add_figure(self, tag: str, figure: Any, global_step: Optional[int] = None, *args, **kwargs):
        wandb = _wandb()
        if figure.__class__.__module__.startswith("plotly"):
            value = wandb.Plotly(figure)
        else:
            value = wandb.Image(figure)
        self._state.log({self._key(tag): value}, step=global_step)

    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None, *args, **kwargs):
        wandb = _wandb()
        self._state.log({self._key(tag): wandb.Html(html.escape(str(text_string)))}, step=global_step)

    def add_mesh(
        self,
        tag: str,
        vertices: Any,
        colors: Any = None,
        faces: Any = None,
        config_dict: Optional[Mapping[str, Any]] = None,
        global_step: Optional[int] = None,
        *args,
        **kwargs,
    ):
        wandb = _wandb()
        objects = []
        vertices_batch = _split_mesh_batch(vertices)
        colors_batch = _split_mesh_batch(colors) if colors is not None else [None] * len(vertices_batch)
        faces_batch = _split_mesh_batch(faces) if faces is not None else [None] * len(vertices_batch)
        for index, verts in enumerate(vertices_batch):
            verts = np.asarray(verts, dtype=np.float32)
            cols = colors_batch[index] if index < len(colors_batch) else None
            face_arr = faces_batch[index] if index < len(faces_batch) else None
            if face_arr is not None:
                objects.append({"type": "mesh", "vertices": verts, "faces": np.asarray(face_arr, dtype=np.int32)})
            else:
                if cols is not None:
                    cols = np.asarray(cols)
                    if cols.ndim == 2 and cols.shape[1] >= 3:
                        points = np.concatenate([verts[:, :3], cols[:, :3]], axis=1)
                    else:
                        points = verts[:, :3]
                else:
                    points = verts[:, :3]
                objects.append(points)
        value = [wandb.Object3D(obj) for obj in objects]
        self._state.log({self._key(tag): value[0] if len(value) == 1 else value}, step=global_step)

    def add_hparams(
        self,
        hparam_dict: Optional[Mapping[str, Any]] = None,
        metric_dict: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        global_step: Optional[int] = None,
        *args,
        **kwargs,
    ):
        if hparam_dict:
            self.run.config.update(dict(hparam_dict), allow_val_change=True)
        if metric_dict:
            prefix = f"{name}/" if name and name != "." else ""
            self.log({f"{prefix}{k}": v for k, v in metric_dict.items()}, step=global_step)

    def watch(self, model: Any, log: str = "all", **kwargs):
        self._state.flush()
        self.run.watch(model, log=log, **kwargs)

    def with_prefix(self, prefix: str):
        child_prefix = f"{self.prefix}/{prefix}" if self.prefix else str(prefix)
        return WandbWriter(self.run, prefix=child_prefix, owns_run=False, state=self._state)

    def flush(self):
        self._state.flush()

    def close(self):
        global _ACTIVE_WRITER, _OWNS_ACTIVE_RUN
        if self.owns_run and self.run is not None:
            self._state.finish()
            if _ACTIVE_WRITER is self:
                _ACTIVE_WRITER = None
                _OWNS_ACTIVE_RUN = False
            self.run = None


def create_wandb_writer(
    logdir: Optional[str] = None,
    *,
    project: str = "RoHM",
    entity: Optional[str] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
    config: Any = None,
    mode: Optional[str] = None,
    enabled: bool = True,
    is_main_process: bool = True,
    resume: Optional[str] = None,
    run_id: Optional[str] = None,
    prefix: str = "",
    job_type: str = "train",
    reinit: bool = False,
    log_code: bool = False,
) -> WandbWriter | NullWandbWriter:
    """Create or reuse the single W&B run for this process."""
    global _ACTIVE_WRITER, _OWNS_ACTIVE_RUN

    if not enabled or not is_main_process:
        return NullWandbWriter()

    wandb_mode = mode or os.environ.get("WANDB_MODE", "offline")
    try:
        wandb = _wandb()
    except ModuleNotFoundError:
        if _is_disabled_mode(wandb_mode):
            return NullWandbWriter()
        raise ModuleNotFoundError(
            "W&B logging requires the 'wandb' package. Install dependencies or set WANDB_MODE=disabled."
        )
    if _ACTIVE_WRITER is not None and getattr(_ACTIVE_WRITER, "run", None) is not None:
        return _ACTIVE_WRITER.with_prefix(prefix) if prefix else _ACTIVE_WRITER

    if wandb.run is not None:
        _ACTIVE_WRITER = WandbWriter(wandb.run, owns_run=False)
        _OWNS_ACTIVE_RUN = False
        return _ACTIVE_WRITER.with_prefix(prefix) if prefix else _ACTIVE_WRITER

    if logdir:
        os.makedirs(logdir, exist_ok=True)

    display_name, stable_run_id = _stable_run_identity(name, run_id, logdir)
    settings = wandb.Settings(save_code=log_code)
    run = wandb.init(
        project=project,
        entity=entity,
        dir=logdir,
        name=display_name,
        group=group or display_name,
        tags=tags,
        notes=notes,
        config=_sanitize_config(config),
        mode=wandb_mode,
        resume=_normalize_resume(resume),
        id=stable_run_id,
        job_type=job_type,
        reinit=reinit,
        settings=settings,
    )
    _ACTIVE_WRITER = WandbWriter(run, owns_run=True)
    _OWNS_ACTIVE_RUN = True
    return _ACTIVE_WRITER.with_prefix(prefix) if prefix else _ACTIVE_WRITER


def get_wandb_writer(prefix: str = "") -> WandbWriter | NullWandbWriter:
    if _ACTIVE_WRITER is None or getattr(_ACTIVE_WRITER, "run", None) is None:
        return NullWandbWriter()
    return _ACTIVE_WRITER.with_prefix(prefix) if prefix else _ACTIVE_WRITER


class WandbKVOutputFormat:
    """Simple KV writer that logs scalar rows to W"""OpenAI-baselines-style KV writer that logs scalar rows to W&B."""B."""

    def __init__(self, dir: str, log_suffix: str = "", project: str = "RoHM", is_main_process: bool = True):
        suffix = log_suffix.strip("-").replace("-", "/")
        self.step = 1
        self.writer = create_wandb_writer(
            dir,
            project=project,
            prefix=suffix,
            is_main_process=is_main_process,
            mode=os.environ.get("WANDB_MODE", "offline"),
        )

    def writekvs(self, kvs: Mapping[str, Any]):
        payload = {}
        for key, value in kvs.items():
            try:
                payload[key] = float(_to_python(value))
            except (TypeError, ValueError):
                continue
        if payload:
            self.writer.log(payload, step=self.step)
        self.step += 1

    def close(self):
        # The shared run is closed by the top-level writer/lifecycle owner.
        return None
