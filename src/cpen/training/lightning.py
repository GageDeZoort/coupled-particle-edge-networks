"""Shared Lightning training helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Union

import lightning as L
import torch
from lightning.fabric.plugins.environments import LightningEnvironment, SLURMEnvironment
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from cpen.utils.callbacks import (
    FeatureMovementCallback,
    ParquetLoggerCallback as _ParquetLoggerCallback,
    TrainingLifecycleCallback,
)
from cpen.utils.log_utils import log_info


class ParquetLoggerCallback(_ParquetLoggerCallback):
    """Parquet logger that also prints val metrics after every validation pass.

    Needed for step-interval validation (``val_check_interval=1``): the base
    logger only prints an epoch line at ``on_train_epoch_end``, so mid-epoch
    val checks would otherwise be silent in the job log.
    """

    _VAL_PRINT_KEYS = (
        "val_loss",
        "val_acc",
        "val_auroc",
        "val_roc_auc",
        "val_node_loss",
        "val_edge_loss",
        "val_edge_acc",
        "val_edge_auroc",
        "val_mean_p_stream",
    )

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)
        if getattr(trainer, "sanity_checking", False):
            return
        if not self._is_rank_zero(trainer):
            return
        parts = [
            f"[val] epoch={trainer.current_epoch}",
            f"step={trainer.global_step}",
        ]
        metrics = trainer.callback_metrics
        for key in self._VAL_PRINT_KEYS:
            value = metrics.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = float(value.detach().cpu())
            else:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
            if value != value:  # NaN
                continue
            parts.append(f"{key}={value:.6g}")
        if len(parts) > 2:
            log_info(" ".join(parts))

PrecisionSetting = Literal["bf16-mixed", "16-mixed", "32-true", "64-true"]
StrategySetting = Union[str, DDPStrategy]


def slurm_ntasks() -> int:
    """Number of Slurm tasks in the current job (1 if not under Slurm)."""
    return int(os.environ.get("SLURM_NTASKS", "1"))


def slurm_ntasks_per_node() -> int | None:
    """Slurm tasks per node when set (required by Lightning's SLURM plugin)."""
    raw = os.environ.get("SLURM_NTASKS_PER_NODE")
    if raw is None:
        return None
    # Slurm may format this as "4" or "4(x2)" for heterogeneous jobs.
    return int(str(raw).split("(")[0])


def slurm_launched_ddp() -> bool:
    """True when ``srun`` launched a multi-task Slurm job for DDP."""
    return slurm_ntasks() > 1 and slurm_ntasks_per_node() is not None


def distributed_world_size(n_gpus: int) -> int:
    """Total GPU count: Slurm task count when multi-task, else ``n_gpus``."""
    tasks = slurm_ntasks()
    return tasks if tasks > 1 else n_gpus


def _trainer_parallel_settings(
    *, n_gpus: int, use_gpu: bool
) -> tuple[int, StrategySetting, int, str]:
    """
    Return (devices, strategy, num_nodes, launch_mode) for Lightning.

    We pass explicit ``cluster_environment`` objects because Lightning's Slurm
    auto-detection activates whenever ``SLURM_NTASKS`` is set (even ``=1``),
    which breaks single-process ``ddp_spawn``.
    """
    if not use_gpu:
        return 1, "auto", 1, "cpu"
    if slurm_ntasks() > 1 and slurm_ntasks_per_node() is None:
        raise RuntimeError(
            f"Detected SLURM_NTASKS={slurm_ntasks()} but SLURM_NTASKS_PER_NODE is unset. "
            "Use #SBATCH --ntasks-per-node=N with 'srun python ...' and "
            "#SBATCH --gpus-per-task=1 (Trainer devices=1 per process)."
        )
    if slurm_launched_ddp():
        # Each srun task is one Lightning process. Slurm remaps that task's
        # GPU to CUDA_VISIBLE_DEVICES=0, so devices is 1, not ntasks-per-node.
        # World size comes from SLURM_NTASKS via SLURMEnvironment.
        visible = torch.cuda.device_count()
        tasks_here = slurm_ntasks_per_node()
        assert tasks_here is not None
        if visible > 1:
            raise RuntimeError(
                f"Slurm DDP: this task sees {visible} GPUs but "
                f"ntasks-per-node={tasks_here}. Use --gpus-per-task=1 so each "
                "srun task gets one GPU, or --ntasks-per-node=1 with "
                f"--gpus-per-task={n_gpus} and a single `python` (no srun)."
            )
        devices = 1
        num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
        strategy = DDPStrategy(cluster_environment=SLURMEnvironment())
        return devices, strategy, num_nodes, "slurm-multitask"
    if n_gpus > 1:
        strategy = DDPStrategy(cluster_environment=LightningEnvironment())
        return n_gpus, strategy, 1, "ddp-spawn"
    return 1, "auto", 1, "single-gpu"


def configure_torch_matmul(*, use_gpu: bool) -> str | None:
    """
    Enable Tensor Core matmuls on GPU (fp32 paths use TF32).

    Returns the matmul precision token logged in run metadata, or None on CPU.
    """
    if not use_gpu:
        return None
    torch.set_float32_matmul_precision("high")
    return "high"


def resolve_precision(*, n_gpus: int, precision: str | None) -> PrecisionSetting:
    """Pick Lightning precision; bf16-mixed on GPU by default, fp32 on CPU."""
    if precision is not None:
        return precision  # type: ignore[return-value]
    return "bf16-mixed" if n_gpus > 0 else "32-true"


def last_checkpoint_callback(
    run_dir: Path, *, every_n_train_steps: int | None = None
) -> ModelCheckpoint:
    """``last.ckpt`` at epoch end, and optionally every ``N`` optimizer steps."""
    kwargs: dict[str, Any] = {
        "dirpath": run_dir,
        "filename": "last",
        "save_last": True,
    }
    if every_n_train_steps is not None and int(every_n_train_steps) > 0:
        kwargs["every_n_train_steps"] = int(every_n_train_steps)
    else:
        kwargs["every_n_epochs"] = 1
    return ModelCheckpoint(**kwargs)


def build_trainer(
    run_dir: Path,
    parquet_path: Path,
    *,
    n_gpus: int = 1,
    max_epochs: int = 100,
    precision: str | None = None,
    log_every_n_steps: int | None = 2500,
    parquet_metadata: dict[str, Any] | None = None,
    track_feature_movement: bool = False,
    feature_movement_interval: int = 500,
    check_val_every_n_epoch: int = 1,
    val_check_interval: float | int | None = None,
    limit_val_batches: float | int | None = None,
    checkpoint_monitor: str = "val_loss",
    checkpoint_mode: str = "min",
    checkpoint_every_n_steps: int | None = None,
) -> L.Trainer:
    """Create a Lightning trainer with DDP, parquet logging, and resume checkpoints."""
    use_gpu = n_gpus > 0
    matmul_precision = configure_torch_matmul(use_gpu=use_gpu)
    resolved_precision = resolve_precision(n_gpus=n_gpus, precision=precision)
    devices, strategy, num_nodes, launch_mode = _trainer_parallel_settings(
        n_gpus=n_gpus, use_gpu=use_gpu
    )
    monitor = str(checkpoint_monitor or "val_loss")
    mode = str(checkpoint_mode or "min")
    if parquet_metadata is not None:
        parquet_metadata["precision"] = resolved_precision
        parquet_metadata["ddp_launch_mode"] = launch_mode
        parquet_metadata["trainer_devices"] = devices
        parquet_metadata["trainer_num_nodes"] = num_nodes
        parquet_metadata["check_val_every_n_epoch"] = check_val_every_n_epoch
        parquet_metadata["checkpoint_monitor"] = monitor
        parquet_metadata["checkpoint_mode"] = mode
        if val_check_interval is not None:
            parquet_metadata["val_check_interval"] = val_check_interval
        if limit_val_batches is not None:
            parquet_metadata["limit_val_batches"] = limit_val_batches
        if matmul_precision is not None:
            parquet_metadata["float32_matmul_precision"] = matmul_precision
        if log_every_n_steps is not None and log_every_n_steps > 0:
            parquet_metadata["log_every_n_steps"] = log_every_n_steps
        if checkpoint_every_n_steps is not None:
            parquet_metadata["checkpoint_every_n_steps"] = int(
                checkpoint_every_n_steps
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        TrainingLifecycleCallback(metadata=parquet_metadata or {}),
        ParquetLoggerCallback(
            parquet_path,
            metadata=parquet_metadata or {},
            log_every_n_steps=log_every_n_steps,
        ),
        last_checkpoint_callback(
            run_dir, every_n_train_steps=checkpoint_every_n_steps
        ),
        ModelCheckpoint(
            dirpath=run_dir,
            filename="best",
            monitor=monitor,
            mode=mode,
            save_top_k=1,
        ),
    ]
    if track_feature_movement:
        callbacks.append(
            FeatureMovementCallback(interval=feature_movement_interval)
        )
    trainer_kwargs: dict[str, Any] = {
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": devices if use_gpu else 1,
        "num_nodes": num_nodes,
        "strategy": strategy,
        "max_epochs": max_epochs,
        "precision": resolved_precision,
        "enable_progress_bar": False,
        "logger": False,
        "callbacks": callbacks,
        "check_val_every_n_epoch": check_val_every_n_epoch,
    }
    if val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = val_check_interval
    if limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = limit_val_batches
    if checkpoint_every_n_steps is not None and int(checkpoint_every_n_steps) > 0:
        # Recreate the stream loader after resume so skip-ahead sees global_step.
        trainer_kwargs["reload_dataloaders_every_n_epochs"] = 1
    return L.Trainer(**trainer_kwargs)


def checkpoint_compatible(ckpt_path: Path, module: torch.nn.Module) -> bool:
    """True if *module* state_dict shapes match the checkpoint (resume-safe)."""
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        return False
    for key, param in module.state_dict().items():
        saved = state_dict.get(key)
        if saved is None:
            continue
        if tuple(saved.shape) != tuple(param.shape):
            return False
    return True


def last_checkpoint_path(run_dir: Path) -> Path | None:
    """Return path to last.ckpt if present."""
    path = run_dir / "last.ckpt"
    return path if path.exists() else None


def best_checkpoint_path(run_dir: Path) -> Path | None:
    """Return path to best.ckpt if present."""
    path = run_dir / "best.ckpt"
    return path if path.exists() else None
