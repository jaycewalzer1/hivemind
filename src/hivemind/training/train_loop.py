import argparse
import glob
from pathlib import Path

from hivemind.paths import PROJECT_ROOT as project_root, TRAINING_OUTPUT_DIR
import polars as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from hivemind.config.train_config import TrainConfig, TrainObjects, rl_train_config
from hivemind.config.main_config import main_config
from hivemind.training.data_loaders import (
    load_parquet_shard,
    StreamingRLDataset,
    infer_input_channels,
)
from hivemind.data.convert_selfplay_data import (
    DEFAULT_Q_VALUE_RATIO,
    DEFAULT_REPLAY_FILES,
    DEFAULT_REPLAY_SELECTION_FRACTION,
    DEFAULT_RL_SAMPLES_PER_SHARD,
    DEFAULT_RL_VALIDATION_FRACTION,
    convert_to_split_parquet,
)

from hivemind.training.schedules import (
    CosineAnnealingSchedule, LinearWarmUp, MomentumSchedule, OneCycleSchedule,
)
from hivemind.architectures.model_config import (
    MODEL_FACTORIES,
    create_model,
    get_model_args,
)
from hivemind.training.trainer_agent import TrainerAgentPytorch, save_torch_state,\
    restore_torch_state, export_to_onnx, get_context
from hivemind.training.train_util import get_metrics
from hivemind.constants import (
    NUM_BUGHOUSE_CHANNELS,
    NUM_CLOCK_BUGHOUSE_CHANNELS,
    NUM_ELAPSED_BUGHOUSE_CHANNELS,
)

AUX_HEAD_KEY_PREFIXES = ("think_gain_head.", "arrivals_head.")


SUPERVISED_WEIGHTS_DIR = TRAINING_OUTPUT_DIR / "weights" / "supervised"
RL_WEIGHTS_DIR = TRAINING_OUTPUT_DIR / "weights" / "rl"
CROSS_BOARD_DEFAULT_BATCH_SIZE = 256
DUAL_STREAM_DEFAULT_BATCH_SIZE = 256
SUPERVISED_EVAL_BATCHES = 64
SUPERVISED_EVAL_INTERVAL_MULTIPLIER = 2


def _resolve_validation_shard(validation_shard: str = None) -> str:
    if validation_shard:
        path = Path(validation_shard).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Validation shard not found: {path}")
        return str(path)

    preferred = project_root / 'data' / 'planes' / 'val' / 'evaluation_shard.parquet'
    if preferred.exists():
        return str(preferred)

    val_dir = Path(main_config['planes_val_dir'])
    parquet_files = sorted(val_dir.glob('*.parquet'))
    if parquet_files:
        return str(parquet_files[0])

    raise FileNotFoundError(
        f"No validation parquet shard found. Checked {preferred} and {val_dir}"
    )


def _resolve_train_evaluation_shard(shard_path: str = None) -> str:
    if shard_path:
        path = Path(shard_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training evaluation shard not found: {path}")
        return str(path)

    preferred = (
        project_root / 'data' / 'planes' / 'train_eval' /
        'evaluation_shard.parquet'
    )
    if preferred.is_file():
        return str(preferred)

    train_dir = Path(main_config['planes_train_dir'])
    parquet_files = sorted(train_dir.glob('*.parquet'))
    if parquet_files:
        return str(parquet_files[0])

    raise FileNotFoundError(
        f"No training evaluation shard found. Checked {preferred} and {train_dir}"
    )


def configure_batch_size(train_config, architecture, batch_size=None):
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")

    original_batch_size = train_config.batch_size
    selected_batch_size = batch_size
    if selected_batch_size is None:
        selected_batch_size = {
            "crossboard-risev33": CROSS_BOARD_DEFAULT_BATCH_SIZE,
            "dualstream-memory-risev33": DUAL_STREAM_DEFAULT_BATCH_SIZE,
        }.get(architecture)
    if selected_batch_size is None:
        return

    train_config.batch_size = selected_batch_size
    train_config.batch_steps = max(
        1,
        round(
            train_config.batch_steps
            * original_batch_size
            / selected_batch_size
        ),
    )
    print(
        f"Training batch size: {train_config.batch_size} "
        f"(evaluate every {train_config.batch_steps} batches)"
    )


def configure_precision(train_config, architecture, precision=None, is_rl=False):
    if precision is None:
        precision = (
            "bf16"
            if is_rl or architecture == "dualstream-memory-risev33"
            else "fp32"
        )
    train_config.mixed_precision = precision
    print(f"Training precision: {precision}")


def configure_supervised_evaluation(train_config):
    train_config.batch_steps *= SUPERVISED_EVAL_INTERVAL_MULTIPLIER
    train_config.eval_batches = SUPERVISED_EVAL_BATCHES
    train_config.full_eval_each_epoch = True


def _checkpoint_progress(checkpoint) -> tuple[int, int]:
    required_keys = {"training_iteration", "evaluation_step"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Checkpoint is missing required progress metadata: {missing}"
        )
    return (
        int(checkpoint["training_iteration"]),
        int(checkpoint["evaluation_step"]),
    )


def get_model_prefix(
    mode: str,
    iteration: int | str = None,
    architecture: str = "risev33",
    val_loss: float = None,
    policy_acc: float = None,
) -> str:
    """Generate model file prefix adhering to option 1 naming scheme."""
    if mode == "rl":
        iter_str = f"it{int(iteration):02d}" if isinstance(iteration, (int, float)) or (isinstance(iteration, str) and iteration.isdigit()) else f"{iteration}"
        if val_loss is not None and policy_acc is not None:
            return f"hivemind-rl-{iter_str}-{architecture}-loss{val_loss:.3f}-p{policy_acc * 100:.1f}"
        return f"hivemind-rl-{iter_str}-{architecture}"
    else:
        if val_loss is not None and policy_acc is not None:
            return f"hivemind-sl-{architecture}-loss{val_loss:.3f}-p{policy_acc * 100:.1f}"
        return f"hivemind-sl-{architecture}"


def save_final_rl_checkpoint(
    model,
    optimizer,
    weights_dir: Path,
    training_iteration: int = None,
    evaluation_step: int = None,
    batch_steps: int = None,
    model_prefix: str = "hivemind-rl-final",
) -> Path:
    """Save the final trainable RL state for the next self-play iteration."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / f"{model_prefix}.tar"
    save_torch_state(
        model,
        optimizer,
        checkpoint_path,
        training_iteration=training_iteration,
        evaluation_step=evaluation_step,
        batch_steps=batch_steps,
    )
    return checkpoint_path


def train_supervised(
    checkpoint_path: str = None,
    train_eval_shard: str = None,
    architecture: str = "risev33",
    batch_size: int = None,
    precision: str = None,
    train_data_dir: str = None,
    validation_shard: str = None,
):
    """Run supervised learning training on human game data."""
    tc = TrainConfig()
    tc.export_dir = f"{TRAINING_OUTPUT_DIR}/"
    tc.weights_dir = str(SUPERVISED_WEIGHTS_DIR)
    tc.use_wdl = True
    tc.use_plys_to_end = True
    tc.policy_loss_factor = 0.978
    configure_supervised_evaluation(tc)
    configure_batch_size(tc, architecture, batch_size)
    configure_precision(tc, architecture, precision)
    print(
        f"Intermediate evaluation: up to {tc.eval_batches} batches per loader "
        f"every {tc.batch_steps} training batches; full evaluation at epoch end"
    )
    checkpoint = None
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    to = TrainObjects()
    to.metrics = get_metrics(tc)

    train_dir = (
        Path(train_data_dir).expanduser().resolve()
        if train_data_dir else Path(main_config['planes_train_dir'])
    )
    train_shards = sorted(str(path) for path in train_dir.glob('*.parquet'))
    tc.nb_parts = len(train_shards)
    if tc.nb_parts == 0:
        raise FileNotFoundError(
            f"No training parquet shards found in {train_dir}"
        )

    # Load validation data
    val_shard = _resolve_validation_shard(validation_shard)
    val_tensors = load_parquet_shard(
        val_shard,
        include_auxiliary=True,
    )
    dataset = TensorDataset(*val_tensors)
    val_data = DataLoader(dataset, batch_size=tc.batch_size, shuffle=False)

    train_eval_tensors = load_parquet_shard(
        _resolve_train_evaluation_shard(train_eval_shard),
        include_auxiliary=True,
    )
    train_eval_data = DataLoader(
        TensorDataset(*train_eval_tensors),
        batch_size=tc.batch_size,
        shuffle=False,
    )

    nb_it_per_epoch = (2**16 * tc.nb_parts) // tc.batch_size
    tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)

    to.lr_schedule = OneCycleSchedule(start_lr=tc.max_lr / 8, max_lr=tc.max_lr, cycle_length=tc.total_it * .3,
                                      cooldown_length=tc.total_it * .6, finish_lr=tc.min_lr)
    to.lr_schedule = LinearWarmUp(to.lr_schedule, start_lr=tc.min_lr, length=tc.total_it / 30)
    to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)

    args = get_model_args(tc, architecture)
    model = create_model(args)

    checkpoint_state = None
    resume_iteration = 0
    if checkpoint is not None:
        checkpoint_state = torch.load(checkpoint, map_location="cpu")
        resume_iteration, tc.k_steps_initial = _checkpoint_progress(
            checkpoint_state
        )

    trainer = TrainerAgentPytorch(
        model,
        val_data,
        tc,
        to,
        use_rtpt=True,
        is_rl=False,
        train_eval_loader=train_eval_data,
    )
    trainer.ordering = train_shards
    if checkpoint_state is not None:
        restore_torch_state(
            model,
            trainer.optimizer,
            checkpoint_state,
        )
        print(
            f"Resuming supervised training from {checkpoint} "
            f"at iteration {resume_iteration}"
        )
    trainer.train(cur_it=resume_iteration)


def rl_parquet_input_channels(parquet_file) -> int:
    """Input channels of an RL Parquet shard: 74 stock, 82 clock planes, 86 v2."""
    first = pl.scan_parquet(parquet_file).select("x").head(1).collect()
    return infer_input_channels(first["x"].to_list())


def checkpoint_input_channels(state_dict) -> int | None:
    """Input channels the checkpoint's input stem expects, if it has one."""
    stem_weight = state_dict.get("body_spatial.0.body.0.weight")
    if stem_weight is not None:
        return int(stem_weight.shape[1])
    shared_stem_weight = state_dict.get("shared_stem.body.0.weight")
    if shared_stem_weight is not None:
        return 2 * int(shared_stem_weight.shape[1])
    return None


def checkpoint_has_aux_heads(state_dict) -> bool:
    """True when the checkpoint carries the spec v2 think_gain/arrivals heads."""
    return any(key.startswith(AUX_HEAD_KEY_PREFIXES) for key in state_dict)


def train_rl(rl_data_dir: str, val_data_dir: str, checkpoint_path: str = None,
             augment_flip: bool = True, architecture: str = "risev33",
             batch_size: int = None, precision: str = None,
             iteration_label: str = None, shuffle_buffer_size: int = 10000,
             resume_training: bool = False, aux_heads: bool | None = None,
             joint_policy_rank: int | None = None,
             joint_policy_loss_factor: float | None = None):
    """
    Run RL training on self-play data.
    
    Args:
        rl_data_dir: Directory containing RL parquet files (converted from binary)
        val_data_dir: Directory containing validation parquet files
        checkpoint_path: Optional path to load model weights from
        augment_flip: If True, use board flip augmentation to double training data
        iteration_label: Optional label for iteration (e.g. "9" or "it09")
        shuffle_buffer_size: Samples retained for shuffling within each shard
        resume_training: Restore optimizer and progress from an interrupted run
        aux_heads: Train the spec v2 think_gain and arrivals heads. None: on
            when the checkpoint has them, or, without a checkpoint, when the
            data has 86 channels. Their losses only use labelled samples.
        joint_policy_rank / joint_policy_loss_factor: override the RL config
            (both default off, see docs in rl_train_config). The joint loss
            needs a model with joint_factors heads AND a factor above 0.
    """
    if shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be positive")

    tc = rl_train_config()
    if joint_policy_rank is not None:
        tc.joint_policy_rank = int(joint_policy_rank)
    if joint_policy_loss_factor is not None:
        tc.joint_policy_loss_factor = float(joint_policy_loss_factor)
    configure_batch_size(tc, architecture, batch_size)
    configure_precision(tc, architecture, precision, is_rl=True)
    to = TrainObjects()
    to.metrics = get_metrics(tc)
    
    # Set export directory and ensure it exists
    tc.export_dir = f"{TRAINING_OUTPUT_DIR}/"
    tc.weights_dir = str(RL_WEIGHTS_DIR)
    weights_dir = RL_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(tc.export_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure ONNX export is enabled
    tc.export_weights = True
    
    # Find all training parquet files
    parquet_files = sorted(glob.glob(str(Path(rl_data_dir) / "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {rl_data_dir}")
    
    print(f"Found {len(parquet_files)} training parquet files in {rl_data_dir}")
    if tc.joint_policy_rank > 0:
        schema_names = set(
            pl.scan_parquet(parquet_files[0]).collect_schema().names()
        )
        if "joint_policy_count" not in schema_names:
            raise ValueError(
                "Joint-policy RL training requires HVM5-converted Parquet data"
            )
        joint_target_count = int(
            pl.scan_parquet(parquet_files)
            .select(pl.col("joint_policy_count").sum())
            .collect()
            .item()
        )
        if joint_target_count == 0:
            raise ValueError(
                "Joint-policy RL training found no HVM5 joint visit targets"
            )
    
    # Load validation data from separate directory
    val_parquet_files = sorted(glob.glob(str(Path(val_data_dir) / "*.parquet")))
    if not val_parquet_files:
        raise ValueError(f"No parquet files found in validation directory {val_data_dir}")
    
    print(f"Found {len(val_parquet_files)} validation parquet files in {val_data_dir}")
    validation_samples = int(
        pl.scan_parquet(val_parquet_files).select(pl.len()).collect().item()
    )
    if validation_samples == 0:
        raise ValueError(f"No validation samples found in {val_data_dir}")

    # The input width follows the data: 74 channels (stock), 82 with clock
    # planes, 86 with clock and elapsed planes; train and validation agree.
    input_channels = rl_parquet_input_channels(parquet_files[0])
    val_input_channels = rl_parquet_input_channels(val_parquet_files[0])
    if val_input_channels != input_channels:
        raise ValueError(
            f"Training data has {input_channels} input channels but "
            f"validation data has {val_input_channels}"
        )

    # Load the checkpoint first: it decides whether the aux heads are trained.
    checkpoint = None
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        # Stage on CPU so a CUDA-saved checkpoint cannot create a second,
        # transient GPU copy during deserialization.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    elif resume_training:
        raise ValueError("resume_training requires checkpoint_path")
    if aux_heads is None:
        aux_heads = (
            checkpoint_has_aux_heads(checkpoint['model_state_dict'])
            if checkpoint is not None
            else input_channels == NUM_ELAPSED_BUGHOUSE_CHANNELS
        )
    tc.aux_head_labels = bool(aux_heads)
    print(f"Aux heads (think_gain, arrivals): {'ON' if aux_heads else 'off'}")

    # Stream validation one shard at a time. Dense dual-policy targets make a
    # fully materialized validation set tens of gigabytes at RL scale.
    val_dataset = StreamingRLDataset(
        val_parquet_files,
        shuffle_files=False,
        augment_flip=augment_flip,
        shuffle_samples=False,
        include_joint_policy=tc.joint_policy_rank > 0,
        include_aux_labels=tc.aux_head_labels,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tc.batch_size,
        num_workers=0,
    )
    augmented_validation_samples = validation_samples * (
        2 if augment_flip else 1
    )
    print(
        f"Validation samples: {augmented_validation_samples} "
        "(streamed from Parquet)"
    )
    
    # Count rows from Parquet metadata so partial shards are handled exactly.
    training_samples = int(
        pl.scan_parquet(parquet_files).select(pl.len()).collect().item()
    )
    if augment_flip:
        training_samples *= 2
    n_train = training_samples
    
    print(f"Board flip augmentation: {'ENABLED' if augment_flip else 'DISABLED'}")
    if augment_flip:
        print("Training data will be doubled through board flip augmentation")
    print(f"Training samples: {n_train}")
    print(f"Training shuffle buffer: {shuffle_buffer_size} samples per shard")
    
    # Create streaming training dataset from all training files
    train_dataset = StreamingRLDataset(
        parquet_files,
        shuffle_files=True,
        shuffle_buffer_size=shuffle_buffer_size,
        augment_flip=augment_flip,
        include_joint_policy=tc.joint_policy_rank > 0,
        include_aux_labels=tc.aux_head_labels,
    )
    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, num_workers=0)
    
    nb_it_per_epoch = max(1, (n_train + tc.batch_size - 1) // tc.batch_size)
    tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)
    tc.nb_parts = len(parquet_files)
    
    print(f"Iterations per epoch: {nb_it_per_epoch}, Total iterations: {tc.total_it}")
    
    # LR schedule: Cosine Annealing with 25% warm-up (as per CrazyAra RL paper)
    # The warm-up helps with context drift in the training data
    cosine_schedule = CosineAnnealingSchedule(min_lr=tc.min_lr, max_lr=tc.max_lr, cycle_length=tc.total_it)
    to.lr_schedule = LinearWarmUp(cosine_schedule, start_lr=tc.min_lr, length=int(tc.total_it * 0.25))
    to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)
    
    # Load model.
    args = get_model_args(
        tc, architecture, input_channels=input_channels, aux_heads=aux_heads
    )
    layout = {
        NUM_CLOCK_BUGHOUSE_CHANNELS: " (clock planes)",
        NUM_ELAPSED_BUGHOUSE_CHANNELS: " (clock and elapsed planes)",
    }.get(input_channels, "")
    print(f"Input planes: {input_channels} channels{layout}")
    model = create_model(args)
    joint_active = (
        getattr(model, "joint_policy_heads", None) is not None
        and tc.joint_policy_loss_factor > 0
    )
    print(
        "Joint-policy loss: "
        + ("ON" if joint_active else
           f"off (joint_policy_rank={tc.joint_policy_rank}, "
           f"joint_policy_loss_factor={tc.joint_policy_loss_factor}; it needs "
           "joint_factors heads and a factor above 0)")
    )
    
    # Optionally load checkpoint
    resume_iteration = None
    if checkpoint is not None:
        checkpoint_channels = checkpoint_input_channels(
            checkpoint['model_state_dict']
        )
        if checkpoint_channels not in (None, input_channels):
            raise RuntimeError(
                f"Checkpoint expects {checkpoint_channels} input channels but "
                f"the RL data has {input_channels}; widen it first "
                "(tools/make_clock_net.py) or use matching data"
            )
        incompatible = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        # Heads the checkpoint lacks start fresh: joint heads, and the aux
        # heads (whose output layers start at 0, so nothing else changes).
        non_joint_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(('joint_policy_heads.', *AUX_HEAD_KEY_PREFIXES))
        ]
        if unexpected or non_joint_missing:
            raise RuntimeError(
                "Checkpoint is incompatible with this architecture: "
                f"missing={non_joint_missing}, unexpected={unexpected}"
            )
        if resume_training:
            resume_iteration, tc.k_steps_initial = _checkpoint_progress(
                checkpoint
            )
            if resume_iteration >= tc.total_it:
                raise ValueError(
                    f"Checkpoint iteration {resume_iteration} has already "
                    f"reached this run's {tc.total_it} iterations"
                )
        else:
            del checkpoint
            checkpoint = None
        print("Checkpoint loaded successfully")
    
    # Train with streaming RL data loader
    trainer = TrainerAgentPytorch(model, val_loader, tc, to, use_rtpt=True, 
                                  is_rl=True, rl_train_loader=train_loader)
    if resume_training:
        restore_torch_state(trainer._model, trainer.optimizer, checkpoint)
        del checkpoint
        print(
            f"Resuming interrupted RL run at iteration {resume_iteration} "
            f"and evaluation step {tc.k_steps_initial}"
        )
    trainer.train(cur_it=resume_iteration)

    # Determine model prefix using Option 1 naming convention: hivemind-rl-it{N}-{arch}-loss{val_loss}-p{policy_acc}
    val_loss = trainer.val_loss_best if trainer.val_loss_best is not None else 0.0
    val_p_acc = trainer.val_p_acc_best if trainer.val_p_acc_best is not None else 0.0
    iter_label = iteration_label if iteration_label else "01"
    model_prefix = get_model_prefix("rl", iter_label, architecture, val_loss, val_p_acc)
    
    # Save a trainable checkpoint before ONNX conversion mutates export state.
    final_checkpoint = save_final_rl_checkpoint(
        model,
        trainer.optimizer,
        weights_dir,
        training_iteration=trainer.cur_it,
        evaluation_step=trainer.k_steps,
        batch_steps=tc.batch_steps,
        model_prefix=model_prefix,
    )
    print(f"\nFinal RL checkpoint saved to {final_checkpoint}")

    print(f"Exporting final model to ONNX with prefix: {model_prefix}...")
    ctx = get_context(tc.context, tc.device_id)
    dummy_input = torch.zeros(1, *args.input_shape).to(ctx)
    export_to_onnx(model, 1, dummy_input, weights_dir, model_prefix, True, True)
    print(f"Final RL ONNX exported under {weights_dir} as {model_prefix}.onnx")


def main():
    parser = argparse.ArgumentParser(description='Train Hivemind neural network')
    parser.add_argument('--mode', type=str, default='sl', choices=['sl', 'rl'],
                        help='Training mode: sl (supervised learning) or rl (reinforcement learning)')
    parser.add_argument(
        '--architecture',
        choices=sorted(MODEL_FACTORIES),
        default='risev33',
        help='Network architecture (default: risev33)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=(
            'Training batch size (defaults: 256 for crossboard-risev33, '
            '256 for dualstream-memory-risev33, otherwise config value)'
        ),
    )
    parser.add_argument(
        '--shuffle-buffer-size',
        type=int,
        default=10000,
        help=(
            'Samples retained for within-shard RL shuffling '
            '(default: 10000; lower values reduce host RAM usage)'
        ),
    )
    parser.add_argument(
        '--precision',
        choices=('fp32', 'bf16'),
        default=None,
        help=(
            'Model execution precision (default: bf16 for RL and '
            'dualstream-memory-risev33, fp32 otherwise)'
        ),
    )
    parser.add_argument('--rl-data-dir', type=str, default=None,
                        help='Preconverted RL training Parquet directory (requires --val-data-dir)')
    parser.add_argument('--val-data-dir', type=str, default=None,
                        help='Preconverted RL validation Parquet directory (requires --rl-data-dir)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument(
        '--resume-training',
        action='store_true',
        help='Restore optimizer and progress from an interrupted RL checkpoint',
    )
    parser.add_argument('--selfplay-dir', type=str,
                        default=str(project_root / 'engine/selfplay_games'),
                        help='Self-play output directory, or its training_data directory, containing HVM chunks')
    parser.add_argument('--replay-dir', type=str, default=None,
                        help='Archived self-play directory used as CrazyAra-style replay memory')
    parser.add_argument('--replay-files', type=int, default=DEFAULT_REPLAY_FILES,
                        help=f'Number of archived HVM chunks to include (default: {DEFAULT_REPLAY_FILES} chunks / 409,600 samples)')
    parser.add_argument('--all-replay', action='store_true',
                        help='Include all archived HVM chunks from replay-dir instead of sampling a subset')
    parser.add_argument(
        '--replay-selection-fraction',
        type=float,
        default=DEFAULT_REPLAY_SELECTION_FRACTION,
        help=f'Newest fraction of replay chunks eligible for selection (default: {DEFAULT_REPLAY_SELECTION_FRACTION})',
    )
    parser.add_argument('--rl-output-dir', type=str, default=None,
                        help='Generated train/validation Parquet directory (default: <selfplay-dir>/rl_data)')
    parser.add_argument(
        '--samples-per-shard',
        type=int,
        default=DEFAULT_RL_SAMPLES_PER_SHARD,
        help=(
            'Samples per generated RL Parquet shard '
            f'(default: {DEFAULT_RL_SAMPLES_PER_SHARD}; lower values reduce host RAM usage)'
        ),
    )
    parser.add_argument(
        '--validation-fraction',
        type=float,
        default=DEFAULT_RL_VALIDATION_FRACTION,
        help=f'Fraction of complete games reserved for validation (default: {DEFAULT_RL_VALIDATION_FRACTION})',
    )
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Deterministic game-level split seed (default: 42)')
    parser.add_argument('--train-eval-shard', type=str, default=None,
                        help='Fixed representative training shard used only for metrics')
    parser.add_argument('--sl-train-data-dir', type=str, default=None,
                        help='Supervised training Parquet directory')
    parser.add_argument('--sl-validation-shard', type=str, default=None,
                        help='Supervised validation Parquet shard')
    parser.add_argument('--iteration', type=str, default=None,
                        help='RL iteration label for model naming (e.g. "9" or "it09", default: auto-detected from selfplay-dir)')
    aux_group = parser.add_mutually_exclusive_group()
    aux_group.add_argument(
        '--aux-heads', dest='aux_heads', action='store_true', default=None,
        help='RL: train the spec v2 think_gain/arrivals heads (default: when '
             'the checkpoint has them, or the data has 86 channels)')
    aux_group.add_argument(
        '--no-aux-heads', dest='aux_heads', action='store_false',
        help='RL: do not train the think_gain/arrivals heads')
    parser.add_argument(
        '--joint-policy-rank', type=int, default=None,
        help='RL: rank of the joint_factors heads (default 0 = none). Heads the '
             'checkpoint lacks start from random init and change engine search.')
    parser.add_argument(
        '--joint-policy-loss-factor', type=float, default=None,
        help='RL: weight of the joint-policy loss (default 0.0 = off)')
    parser.add_argument(
        '--q-value-ratio',
        type=float,
        default=DEFAULT_Q_VALUE_RATIO,
        help=f'Ratio of root Q-value to mix with game outcome (default: {DEFAULT_Q_VALUE_RATIO})',
    )
    
    args = parser.parse_args()

    if args.mode == 'sl':
        train_supervised(
            checkpoint_path=args.checkpoint,
            train_eval_shard=args.train_eval_shard,
            architecture=args.architecture,
            batch_size=args.batch_size,
            precision=args.precision,
            train_data_dir=args.sl_train_data_dir,
            validation_shard=args.sl_validation_shard,
        )
    else:
        if args.rl_data_dir or args.val_data_dir:
            if not args.rl_data_dir or not args.val_data_dir:
                parser.error('--rl-data-dir and --val-data-dir must be provided together')
            train_rl(
                args.rl_data_dir,
                args.val_data_dir,
                checkpoint_path=args.checkpoint,
                architecture=args.architecture,
                batch_size=args.batch_size,
                precision=args.precision,
                iteration_label=args.iteration,
                shuffle_buffer_size=args.shuffle_buffer_size,
                resume_training=args.resume_training,
                aux_heads=args.aux_heads,
                joint_policy_rank=args.joint_policy_rank,
                joint_policy_loss_factor=args.joint_policy_loss_factor,
            )
        else:
            selfplay_path = Path(args.selfplay_dir).expanduser().resolve()
            has_training_data_dir = (selfplay_path / "training_data").is_dir()
            hvm_path = selfplay_path / "training_data" if has_training_data_dir else selfplay_path
            default_output_root = (
                selfplay_path
                if has_training_data_dir or selfplay_path.name != "training_data"
                else selfplay_path.parent
            )
            output_path = (
                Path(args.rl_output_dir).expanduser().resolve()
                if args.rl_output_dir
                else default_output_root / "rl_data"
            )
            replay_path = None
            if args.replay_dir:
                replay_root = Path(args.replay_dir).expanduser().resolve()
                replay_path = (
                    replay_root / "training_data"
                    if (replay_root / "training_data").is_dir()
                    else replay_root
                )
            replay_files = None if args.all_replay else args.replay_files
            train_path, val_path, _, _ = convert_to_split_parquet(
                hvm_path,
                output_path,
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
                replay_input_dir=replay_path,
                replay_files=replay_files,
                replay_selection_fraction=args.replay_selection_fraction,
                q_value_ratio=args.q_value_ratio,
                samples_per_shard=args.samples_per_shard,
            )
            # Auto-detect iteration label from selfplay_path if not explicitly provided
            iter_label = args.iteration
            if not iter_label:
                dir_name = selfplay_path.name if selfplay_path.name != "training_data" else selfplay_path.parent.name
                if "iteration-" in dir_name:
                    iter_label = dir_name.split("iteration-")[-1]
                elif "it" in dir_name:
                    iter_label = dir_name.split("it")[-1]

            train_rl(
                str(train_path),
                str(val_path),
                checkpoint_path=args.checkpoint,
                architecture=args.architecture,
                batch_size=args.batch_size,
                precision=args.precision,
                iteration_label=iter_label,
                shuffle_buffer_size=args.shuffle_buffer_size,
                resume_training=args.resume_training,
                aux_heads=args.aux_heads,
                joint_policy_rank=args.joint_policy_rank,
                joint_policy_loss_factor=args.joint_policy_loss_factor,
            )


if __name__ == "__main__":
    main()
