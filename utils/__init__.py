from .dataset import get_dataloaders
from .transforms import get_train_transforms, get_test_transforms
from .metrics import AverageMeter, compute_accuracy, ConfusionMatrixTracker, EarlyStopping
from .eda import run_eda
from .profiler import Timer, ModelProfiler, benchmark_inference, compare_model_complexities
from .seed import seed_everything, seed_worker, calculate_mean_std, format_mean_std

__all__ = [
    "get_dataloaders",
    "get_train_transforms",
    "get_test_transforms",
    "AverageMeter",
    "compute_accuracy",
    "ConfusionMatrixTracker",
    "EarlyStopping",
    "plot_training_curves",
    "plot_ablation_summary",
    "run_eda",
    "Timer",
    "ModelProfiler",
    "benchmark_inference",
    "compare_model_complexities",
    "seed_everything",
    "seed_worker",
    "calculate_mean_std",
    "format_mean_std",
]


def __getattr__(name):
    if name in {"plot_training_curves", "plot_ablation_summary"}:
        from .visualization import plot_training_curves, plot_ablation_summary
        return {
            "plot_training_curves": plot_training_curves,
            "plot_ablation_summary": plot_ablation_summary,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
