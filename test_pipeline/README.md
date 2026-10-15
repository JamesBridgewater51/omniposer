# Test Pipeline - Refactored Architecture

This package provides a modular, extensible test pipeline for RoHM motion reconstruction evaluation.

## Design Principles

- **High cohesion**: Each module has a single, well-defined responsibility
- **Low coupling**: Modules communicate through clean interfaces, minimizing dependencies
- **Extensibility**: Easy to add new datasets, visualizers, baselines, or analysis methods
- **Backward Compatibility**: Original inference logic preserved; hook-based system for baseline comparison

## Architecture Overview

```
test_pipeline/
├── __init__.py              # Package exports
├── config.py                # Configuration handling and validation
├── runner.py                # Re-exports TestRunner and LegacyTestRunner
├── runner_hooks.py          # Hook-based TestRunner (default)
├── runner_legacy.py         # Original TestRunner (backward compat)
├── baselines/               # Baseline comparison framework
│   ├── __init__.py          # Package exports with get_baseline()
│   ├── registry.py          # BaselineRegistry singleton
│   ├── spec.py              # BaselineSpec frozen dataclass
│   ├── context.py           # InferenceContext shared state
│   ├── hooks/
│   │   ├── __init__.py
│   │   └── base.py          # Abstract hook base classes
│   └── implementations/
│       ├── __init__.py      # Triggers registration on import
│       └── ours.py          # "ours" baseline with full hooks
├── inference/
│   ├── __init__.py
│   ├── batch_processor.py       # Original BatchProcessor
│   ├── batch_processor_hooks.py # Hook-based BatchProcessorWithHooks
│   ├── sliding_window.py        # Window decomposition/fusion
│   ├── normalization.py         # NormalizationStats
│   └── parallel.py              # ParallelExecutor
├── visualization/
│   └── ...
└── analysis/
    └── ...
```

## Baseline Comparison Framework

The baseline system enables fair comparison between different methods using a registry-based hook architecture.

### Core Components

1. **BaselineRegistry** (`registry.py`): Singleton registry for baseline registration/retrieval
2. **BaselineSpec** (`spec.py`): Frozen dataclass defining baseline capabilities and hooks
3. **InferenceContext** (`context.py`): Shared state container passed through all hooks
4. **Hook ABCs** (`hooks/base.py`): Abstract base classes for the four hook types

### Hook Types

```python
PreprocessingHook:
  - prepare_conditioning(batch, ctx) → (traj_batch, pose_batch)
  - apply_masking(traj_batch, pose_batch, masking_spec, ctx) → (masked_traj, masked_pose)

InferenceHook:
  - run_trajnet(traj_batch, ctx) → traj_output
  - inject_trajectory(pose_batch, traj_output, ctx) → updated_pose_batch
  - run_posenet(pose_batch, ctx) → pose_output

PostprocessingHook:
  - reconstruct_repr_dict(pose_output, ctx) → repr_dict
  - recover_world_frame(repr_dict, batch, ctx) → (joints, vertices)

MetricsHook:
  - get_metric_names() → List[str]
  - compute_metrics(pred_joints, gt_joints, repr_dict_rec, repr_dict_clean, ctx) → Dict[str, float]
```

### Execution Flow

```
TestRunner.run()
    │
    ├─► BatchProcessorWithHooks.process_batch(batch)
    │       │
    │       ├─► PreprocessingHook.prepare_conditioning()
    │       │
    │       ├─► [Sliding Window Orchestration - shared infrastructure]
    │       │       │
    │       │       ├─► PreprocessingHook.apply_masking()
    │       │       ├─► InferenceHook.run_trajnet()
    │       │       ├─► InferenceHook.inject_trajectory()
    │       │       └─► InferenceHook.run_posenet()
    │       │
    │       └─► PostprocessingHook.reconstruct_repr_dict()
    │
    └─► MetricsHook.compute_metrics()
```

### Adding a New Baseline

```python
# test_pipeline/baselines/implementations/my_baseline.py
from ..registry import BaselineRegistry
from ..spec import BaselineSpec
from ..hooks.base import PreprocessingHook, InferenceHook, PostprocessingHook, MetricsHook

class MyPreprocessingHook(PreprocessingHook):
    def prepare_conditioning(self, batch, ctx):
        # Custom conditioning logic
        ...
    
    def apply_masking(self, traj_batch, pose_batch, masking_spec, ctx):
        # Custom masking logic
        ...

# ... implement other hooks ...

@BaselineRegistry.register('my_baseline')
def create_my_baseline_spec() -> BaselineSpec:
    return BaselineSpec(
        name='my_baseline',
        compatible_mask_schemes=['rohm_upper_body_xyz', ...],
        default_mask_scheme='rohm_upper_body_xyz',
        preprocessing_hook_cls=MyPreprocessingHook,
        inference_hook_cls=MyInferenceHook,
        postprocessing_hook_cls=MyPostprocessingHook,
        metrics_hook_cls=MyMetricsHook,
        # ... other spec fields ...
    )
```

Then import in `implementations/__init__.py`:
```python
from . import ours
from . import my_baseline  # Add this line
```

### Running with Different Baselines

```bash
# Default (ours)
python test.py logdir=runs/my_run

# Specific baseline
python test.py logdir=runs/my_run baseline=my_baseline
```

### Key Design Decisions

1. **Our method IS a baseline**: `OursBaseline` implements all hooks, no special-casing
2. **No fallbacks**: All hooks required - explicit over implicit
3. **Train config drives capability**: Registries and model configs from training
4. **Masking uses actions**: Test masking uses action names from cfg_files/conf/masking
5. **Sliding window as infrastructure**: Shared across all baselines, not delegated to hooks

## Module Responsibilities

### `config.py`
- `TestConfig`: Dataclass for test configuration
- `apply_mpi_config()`: Apply MPI rank/world_size settings
- `validate_test_config()`: Validate configuration completeness
- `save_test_config()`: Persist configuration to YAML

### `runner.py` - TestRunner
Main orchestrator that:
- Lazily initializes dataset, models, diffusions
- Coordinates batch processing via `ParallelExecutor`
- Triggers visualization via `BaseVisualizer` subclasses
- Computes and saves metrics via `MotionEvaluator`

### `inference/batch_processor.py` - BatchProcessor
Core inference logic:
- Prepares batch conditioning and masking
- Runs TrajNet → PoseNet pipeline
- Handles sliding window for long sequences
- Reconstructs representation dictionaries

### `inference/sliding_window.py` - SlidingWindowProcessor
Sliding window utilities:
- `split_into_windows()`: Decompose sequence into overlapping windows
- `fuse_windows()`: Blend window outputs with overlap weighting
- Root XY initialization for each window

### `inference/parallel.py` - ParallelExecutor
Execution strategies:
- `_execute_single_process()`: Sequential processing
- `_execute_multi_process()`: GPU worker pool with queues

### `visualization/base.py` - BaseVisualizer
Abstract interface:
- `setup()`: Initialize rendering resources
- `visualize(data: VisualizationData)`: Render frame(s)
- `cleanup()`: Release resources

### `visualization/registry.py` - VisualizerRegistry
Factory pattern:
- `@VisualizerRegistry.register('dataset_name')`: Decorator for registration
- `get_visualizer()`: Factory function to create appropriate visualizer

### `analysis/debug_analysis.py` - DebugAnalyzer
Per-sequence analysis:
- Error distribution (MPJPE, per-joint, per-frame)
- Motion dynamics (velocity, acceleration)
- Sliding window boundary effects
- Outlier clustering

### `analysis/rotation_invariance.py` - RotationInvarianceTester
Robustness testing:
- Apply Z-axis rotations to inputs
- Measure reconstruction consistency across angles

## Usage

### Basic Usage
```bash
# Using new entry point
python test_new.py logdir=runs/both_example_run

# With specific checkpoint
python test_new.py logdir=runs/both_example_run \
    model_path_trajnet=path/to/trajnet.safetensors \
    model_path_posenet=path/to/posenet.safetensors

# Debug dataset
python test_new.py logdir=runs/both_example_run dataset=debug
```

### Masking Actions
```bash
# Use a masking action defined in cfg_files/conf/masking/*.yaml
python test.py logdir=runs/both_example_run test_mask_action=egoexo_prior
```

### Programmatic Usage
```python
from test_pipeline import TestRunner
from omegaconf import OmegaConf

train_config = OmegaConf.load("runs/my_run/config.yaml")
test_config = OmegaConf.load("cfg_files/conf/test/config.yaml")

runner = TestRunner(train_config, test_config)
metrics = runner.run()
print(metrics)
```

### Adding a New Visualizer
```python
from test_pipeline.visualization import BaseVisualizer, VisualizerRegistry, VisualizationData

@VisualizerRegistry.register('my_dataset')
class MyDatasetVisualizer(BaseVisualizer):
    def setup(self):
        # Initialize rendering resources
        pass
    
    def visualize(self, data: VisualizationData):
        # Render visualization
        pass
    
    def cleanup(self):
        # Release resources
        pass
```

## Comparison with Original `test.py`

| Aspect | Original `test.py` | Refactored `test_pipeline/` |
|--------|-------------------|----------------------------|
| Lines of Code | ~3700 (single file) | ~1500 (distributed) |
| Visualization | Inline functions | Plugin-based registry |
| Sliding Window | Embedded in `process_one_batch` | Dedicated `SlidingWindowProcessor` |
| Parallelism | Inline multi-process code | `ParallelExecutor` abstraction |
| Configuration | Mixed with logic | Dedicated `config.py` |
| Extensibility | Requires modifying main file | Add new modules |

## Migration Guide

The original `test.py` is preserved for backward compatibility. To migrate:

1. Use `test_new.py` as entry point
2. Same Hydra config structure works
3. Results are saved in same format

For custom modifications:
- Extend `BatchProcessor` for custom inference logic
- Subclass `BaseVisualizer` for new datasets
- Add analysis methods to `DebugAnalyzer`
