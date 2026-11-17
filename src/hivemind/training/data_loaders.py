
"""
Data loading utilities for parquet files
"""
import numpy as np
import polars as pl
import torch
from hivemind.constants import (
    CLOCK_CHANNEL_OFFSET,
    ELAPSED_CHANNEL_OFFSET,
    NUM_BUGHOUSE_CHANNELS,
    NUM_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_CLOCK_BUGHOUSE_CHANNELS_PER_BOARD,
    NUM_ELAPSED_BUGHOUSE_CHANNELS_PER_BOARD,
    board_channels_from_total,
)


# Constants matching the Python representation
NB_INPUT_CHANNELS = NUM_BUGHOUSE_CHANNELS
NB_BUGHOUSE_BOARD_CHANNELS = NUM_BUGHOUSE_CHANNELS_PER_BOARD
BOARD_SIZE = 8
NB_INPUT_VALUES = NB_INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE
NB_POLICY_CHANNELS = 73
NB_POLICY_VALUES = NB_POLICY_CHANNELS * BOARD_SIZE * BOARD_SIZE  # 4672
NB_JOINT_POLICY_VALUES = NB_POLICY_VALUES + 1  # dedicated pass action
MAX_JOINT_POLICY_ENTRIES = 256
NB_ARRIVAL_VALUES = 20
ARRIVALS_PER_BOARD = NB_ARRIVAL_VALUES // 2  # ours + theirs pockets x 5 piece types
AUX_LABEL_COLUMNS = (
    'think_gain_a',
    'think_gain_b',
    'think_gain_mask_a',
    'think_gain_mask_b',
    'arrivals',
    'arrivals_mask',
)


# Spec v3 (WP-D): exact clock scalars written by the converter from the
# recorder's sidecar; the loader rebuilds planes 37-42 of each board block from
# them instead of the uint8 planes (clocks on a 1.18 s grid, exp planes 0 above
# 62 s). Bits of clock_scalars_mask: 1 = clocks present, 2 = elapsed present.
CLOCK_SCALAR_COLUMNS = ('clocks_ms', 'elapsed_ms', 'clock_scalars_mask', 'team')
CLOCKS_MASK_BIT = 1
ELAPSED_MASK_BIT = 2


def clock_plane_tensors(remaining_ms: torch.Tensor):
    """(min(t/300, 1), exp(-t/10)), t = max(ms, 0) / 1000, as planes.cc."""
    t = remaining_ms.double().clamp_min(0.0) / 1000.0
    return (t / 300.0).clamp_max(1.0).float(), torch.exp(-t / 10.0).float()


def elapsed_plane_tensors(elapsed_ms: torch.Tensor):
    """(min(e/10, 1), 1 - exp(-e/2)), e = max(ms, 0) / 1000, as planes.cc."""
    e = elapsed_ms.double().clamp_min(0.0) / 1000.0
    return (e / 10.0).clamp_max(1.0).float(), (1.0 - torch.exp(-e / 2.0)).float()


def rebuild_clock_planes(x: torch.Tensor, clocks_ms: torch.Tensor,
                         elapsed_ms: torch.Tensor, mask: torch.Tensor,
                         team: torch.Tensor) -> torch.Tensor:
    """Overwrite the clock (37-40) and elapsed (41-42) planes of both board
    blocks of normalized planes x (N, C, 8, 8) from exact scalars, in place.

    clocks_ms (N, 4): A_white A_black B_white B_black; elapsed_ms (N, 2): A, B;
    mask (N,): CLOCKS_MASK_BIT / ELAPSED_MASK_BIT; team (N,): 0 = the sample's
    team plays white on board A (T1), 1 = black. Our seat on A is the team's
    colour there, on B the other colour (the partner). Rows without the bit
    keep their planes. Other widths (37 per board) are returned unchanged.
    """
    per_board = x.shape[1] // 2
    if per_board < NUM_CLOCK_BUGHOUSE_CHANNELS_PER_BOARD or len(x) == 0:
        return x
    mask = mask.long()
    white = team.long() == 0
    clocks = clocks_ms.double()
    ours_a = torch.where(white, clocks[:, 0], clocks[:, 1])
    theirs_a = torch.where(white, clocks[:, 1], clocks[:, 0])
    ours_b = torch.where(white, clocks[:, 3], clocks[:, 2])
    theirs_b = torch.where(white, clocks[:, 2], clocks[:, 3])
    rows = (mask & CLOCKS_MASK_BIT) != 0
    if bool(rows.any()):
        for base, ours, theirs in ((0, ours_a, theirs_a), (per_board, ours_b, theirs_b)):
            c = base + CLOCK_CHANNEL_OFFSET
            values = (*clock_plane_tensors(ours), *clock_plane_tensors(theirs))
            for k, value in enumerate(values):
                x[rows, c + k] = value[rows].view(-1, 1, 1).expand(-1, 8, 8).to(x.dtype)
    rows = (mask & ELAPSED_MASK_BIT) != 0
    if per_board >= NUM_ELAPSED_BUGHOUSE_CHANNELS_PER_BOARD and bool(rows.any()):
        for base, column in ((0, 0), (per_board, 1)):
            c = base + ELAPSED_CHANNEL_OFFSET
            for k, value in enumerate(elapsed_plane_tensors(elapsed_ms[:, column])):
                x[rows, c + k] = value[rows].view(-1, 1, 1).expand(-1, 8, 8).to(x.dtype)
    return x


def _apply_exact_clocks(df, x_tensor: torch.Tensor) -> torch.Tensor:
    """rebuild_clock_planes from a shard's clock-scalar columns, when present."""
    if not all(column in df.columns for column in CLOCK_SCALAR_COLUMNS) or not len(df):
        return x_tensor
    mask = torch.tensor(df['clock_scalars_mask'].to_numpy(), dtype=torch.long)
    if not bool(mask.any()):
        return x_tensor
    clocks = torch.from_numpy(_decode_fixed_width_binary_column(
        df['clocks_ms'].to_list(), dtype=np.float32, width=4).copy())
    elapsed = torch.from_numpy(_decode_fixed_width_binary_column(
        df['elapsed_ms'].to_list(), dtype=np.float32, width=2).copy())
    team = torch.tensor(df['team'].to_numpy(), dtype=torch.long)
    return rebuild_clock_planes(x_tensor, clocks, elapsed, mask, team)


def _sample_weight_tensor(df, sample_count: int) -> torch.Tensor:
    """The converter's sample_weight column (spec v3), or ones."""
    if 'sample_weight' not in df.columns:
        return torch.ones(sample_count)
    return torch.tensor(df['sample_weight'].to_numpy(), dtype=torch.float32)


def infer_input_channels(encoded_planes_column) -> int:
    """Input channels of a shard's planes: 74 (default), 82 (clock planes) or 86 (v2).

    An empty column keeps the default. Every row must have the same width,
    which _decode_fixed_width_binary_column checks.
    """
    if not encoded_planes_column:
        return NB_INPUT_CHANNELS
    values = len(encoded_planes_column[0])
    channels, remainder = divmod(values, BOARD_SIZE * BOARD_SIZE)
    if remainder:
        raise ValueError(f"Plane payload of {values} bytes is not whole 8x8 planes")
    board_channels_from_total(channels)
    return channels


def _decode_plane_bytes(encoded_planes) -> np.ndarray:
    planes = np.frombuffer(encoded_planes, dtype=np.uint8)
    if planes.size != NB_INPUT_VALUES:
        raise ValueError(
            f"Expected {NB_INPUT_VALUES} input values, found {planes.size}"
        )
    return planes


def _decode_fixed_width_binary_column(values, dtype, width) -> np.ndarray:
    if not values:
        return np.empty((0, width), dtype=dtype)

    expected_nbytes = width * np.dtype(dtype).itemsize
    for idx, value in enumerate(values):
        if len(value) != expected_nbytes:
            raise ValueError(
                f"Invalid binary payload size at row {idx}: expected {expected_nbytes}, found {len(value)}"
            )

    joined = b"".join(values)
    return np.frombuffer(joined, dtype=dtype).reshape(len(values), width)


def _normalize_input_planes(x_tensor: torch.Tensor) -> torch.Tensor:
    return x_tensor / 255.0


def flip_bughouse_sample(x, policy_a, policy_b):
    """
    Apply bughouse board flip augmentation by swapping the two boards.
    
    In bughouse, due to symmetry, we can swap Board A and Board B to create
    an equivalent position from the partner's perspective. This effectively
    doubles the training data.
    
    IMPORTANT: When swapping boards, we must also mirror the moves in each policy
    because the board perspective changes (like viewing from the opposite side).
    
    Args:
        x: Input planes (C, 8, 8) where C depends on NUM_BUGHOUSE_CHANNELS
        policy_a: Policy distribution for Board A (4672,)
        policy_b: Policy distribution for Board B (4672,)
    
    Returns:
        Tuple of (flipped_x, flipped_policy_a, flipped_policy_b) where:
        - Board A and B channels are swapped
        - Policy distributions are swapped AND mirrored
    """
    # Clone to avoid modifying originals
    flipped_x = x.clone()
    
    # Swap the board channel blocks (Board A ↔ Board B). Each block is half
    # the input: 37 channels, or 41 with clock planes.
    total_channels = x.shape[0]
    offset = board_channels_from_total(total_channels)
    flipped_x[:offset] = x[offset:total_channels]
    flipped_x[offset:total_channels] = x[:offset]
    
    # Swap the policy distributions (NO MIRRORING - just swap)
    # What was Board B becomes Board A -> use policy_b as-is
    # What was Board A becomes Board B -> use policy_a as-is
    # The boards are not being flipped vertically, just swapped, so moves stay the same
    flipped_policy_a = policy_b.clone()
    flipped_policy_b = policy_a.clone()
    
    return flipped_x, flipped_policy_a, flipped_policy_b


def flip_aux_labels(think_gain, think_gain_mask, arrivals, arrivals_mask):
    """Board-swap augmentation for the spec v2 aux labels.

    Swapping boards A and B swaps the two think_gain entries and the two board
    halves of arrivals (ours A, theirs A | ours B, theirs B): the B block's
    "ours" is the partner, which becomes our seat on the new board A.
    """
    flipped_arrivals = torch.cat(
        (arrivals[ARRIVALS_PER_BOARD:], arrivals[:ARRIVALS_PER_BOARD])
    )
    return (
        think_gain.flip(0),
        think_gain_mask.flip(0),
        flipped_arrivals,
        arrivals_mask.clone(),
    )


def load_parquet_shard(file_path, include_auxiliary=False):
    """
    Loads a single supervised learning parquet shard and converts it to PyTorch tensors.
    x: uniformly quantized board planes (74, 8, 8)
    y_value: game outcome
    y_policy_idx: tuple containing (move_index, ...)
    """
    required_columns = ['x', 'y_value', 'y_policy_idx']
    if include_auxiliary:
        required_columns.append('y_plys_to_end')

    df = pl.read_parquet(file_path, columns=required_columns)

    x_values = df['x'].to_list()
    input_channels = infer_input_channels(x_values)
    x_array = _decode_fixed_width_binary_column(
        x_values,
        dtype=np.uint8,
        width=input_channels * BOARD_SIZE * BOARD_SIZE,
    )
    x_tensor = torch.from_numpy(x_array.copy()).float().view(
        -1, input_channels, BOARD_SIZE, BOARD_SIZE
    )
    x_tensor = _normalize_input_planes(x_tensor)

    y_val_tensor = torch.tensor(df['y_value'].to_list(), dtype=torch.float32)
    y_pol_tensor = torch.tensor(df['y_policy_idx'].to_list(), dtype=torch.long)

    if include_auxiliary:
        if 'y_plys_to_end' not in df.columns:
            raise ValueError(
                "Moves-left training requires a y_plys_to_end column; regenerate this shard"
            )
        if not torch.all((y_val_tensor == -1) | (y_val_tensor == 0) | (y_val_tensor == 1)):
            raise ValueError("WDL training requires y_value labels in {-1, 0, 1}")
        wdl_tensor = (y_val_tensor + 1).long()
        plys_tensor = torch.tensor(
            df['y_plys_to_end'].to_list(), dtype=torch.float32
        ).clamp_(0, 100) / 100.0
        return x_tensor, y_val_tensor, y_pol_tensor, wdl_tensor, plys_tensor

    return x_tensor, y_val_tensor, y_pol_tensor


def load_rl_parquet_shard(file_path, include_joint_policy=False,
                          include_aux_labels=False, include_sample_weights=False,
                          exact_clocks=True):
    """
    Loads a single RL/self-play parquet shard and converts it to PyTorch tensors.
    
    RL data format (from C++ self-play):
    - x: bytes (NB_INPUT_VALUES uint8 planes)
    - policy_a: bytes (4672 float32 dense policy distribution for board A)
    - policy_b: bytes (4672 float32 dense policy distribution for board B)
    - y_value: float (game outcome)
    - y_wdl: int in {0, 1, 2}
    - y_moves_left: remaining team decisions
    
    With include_aux_labels, four more tensors follow (after the joint ones):
    think_gain (N, 2), think_gain_mask (N, 2), arrivals (N, 20) and
    arrivals_mask (N,), all float32. A shard without those columns (converted
    from HVM3-HVM5 chunks before spec v2) gives zero labels and zero masks.
    With include_sample_weights, one more tensor ends the tuple: the per-sample
    weight (N,) float32 (spec v3; ones when the shard has no column).
    exact_clocks: rebuild planes 37-42 from the shard's exact clock scalars
    when it has them (rebuild_clock_planes); stock shards are unaffected.

    Returns:
        x_tensor: (N, C, 8, 8) float32 input planes, C = 74, 82 or 86
        y_val_tensor: (N,) float32 value targets
        policy_a_tensor: (N, 4672) float32 policy distribution for board A
        policy_b_tensor: (N, 4672) float32 policy distribution for board B
    """
    columns = ['x', 'policy_a', 'policy_b', 'y_value', 'y_wdl', 'y_moves_left']
    joint_columns = [
        'joint_policy_a',
        'joint_policy_b',
        'joint_policy_probability',
        'joint_policy_count',
    ]
    available_columns = set(pl.scan_parquet(file_path).collect_schema().names())
    has_joint_policy = all(column in available_columns for column in joint_columns)
    if include_joint_policy and has_joint_policy:
        columns.extend(joint_columns)
    has_aux_labels = all(column in available_columns for column in AUX_LABEL_COLUMNS)
    if include_aux_labels and has_aux_labels:
        columns.extend(AUX_LABEL_COLUMNS)
    if exact_clocks and all(column in available_columns for column in CLOCK_SCALAR_COLUMNS):
        columns.extend(c for c in CLOCK_SCALAR_COLUMNS if c not in columns)
    if include_sample_weights and 'sample_weight' in available_columns:
        columns.append('sample_weight')
    df = pl.read_parquet(file_path, columns=columns)

    x_values = df['x'].to_list()
    input_channels = infer_input_channels(x_values)
    x_array = _decode_fixed_width_binary_column(
        x_values,
        dtype=np.uint8,
        width=input_channels * BOARD_SIZE * BOARD_SIZE,
    )
    x_tensor = torch.from_numpy(x_array.copy()).float().view(
        -1, input_channels, BOARD_SIZE, BOARD_SIZE
    )
    x_tensor = _normalize_input_planes(x_tensor)
    if exact_clocks:
        x_tensor = _apply_exact_clocks(df, x_tensor)
    
    # Process Y_Value
    y_val_tensor = torch.tensor(df['y_value'].to_list(), dtype=torch.float32)
    
    policy_a_array = _decode_fixed_width_binary_column(
        df['policy_a'].to_list(),
        dtype=np.float32,
        width=NB_POLICY_VALUES,
    )
    policy_b_array = _decode_fixed_width_binary_column(
        df['policy_b'].to_list(),
        dtype=np.float32,
        width=NB_POLICY_VALUES,
    )
    policy_a_tensor = torch.tensor(policy_a_array.copy(), dtype=torch.float32)
    policy_b_tensor = torch.tensor(policy_b_array.copy(), dtype=torch.float32)
    wdl_tensor = torch.tensor(df['y_wdl'].to_list(), dtype=torch.long)
    moves_left_tensor = torch.tensor(
        df['y_moves_left'].to_list(), dtype=torch.float32
    ).clamp_(0, 100) / 100.0

    base = (
        x_tensor,
        y_val_tensor,
        policy_a_tensor,
        policy_b_tensor,
        wdl_tensor,
        moves_left_tensor,
    )
    sample_count = len(x_tensor)
    aux = ()
    if include_aux_labels:
        aux = _aux_label_tensors(df, sample_count, has_aux_labels)
    if include_sample_weights:
        aux = aux + (_sample_weight_tensor(df, sample_count),)
    if not include_joint_policy:
        return base + aux

    if has_joint_policy:
        joint_a_array = _decode_fixed_width_binary_column(
            df['joint_policy_a'].to_list(),
            dtype=np.uint16,
            width=MAX_JOINT_POLICY_ENTRIES,
        )
        joint_b_array = _decode_fixed_width_binary_column(
            df['joint_policy_b'].to_list(),
            dtype=np.uint16,
            width=MAX_JOINT_POLICY_ENTRIES,
        )
        joint_probability_array = _decode_fixed_width_binary_column(
            df['joint_policy_probability'].to_list(),
            dtype=np.float32,
            width=MAX_JOINT_POLICY_ENTRIES,
        )
        joint_count = torch.tensor(
            df['joint_policy_count'].to_list(), dtype=torch.long
        )
    else:
        joint_a_array = np.zeros(
            (sample_count, MAX_JOINT_POLICY_ENTRIES), dtype=np.uint16
        )
        joint_b_array = np.zeros_like(joint_a_array)
        joint_probability_array = np.zeros(
            (sample_count, MAX_JOINT_POLICY_ENTRIES), dtype=np.float32
        )
        joint_count = torch.zeros(sample_count, dtype=torch.long)

    return base + (
        torch.from_numpy(joint_a_array.astype(np.int64)),
        torch.from_numpy(joint_b_array.astype(np.int64)),
        torch.from_numpy(joint_probability_array.copy()),
        joint_count,
    ) + aux


def _aux_label_tensors(df, sample_count, has_aux_labels):
    """(think_gain, think_gain_mask, arrivals, arrivals_mask) float32 tensors."""
    if not has_aux_labels:
        return (
            torch.zeros(sample_count, 2),
            torch.zeros(sample_count, 2),
            torch.zeros(sample_count, NB_ARRIVAL_VALUES),
            torch.zeros(sample_count),
        )
    think_gain = torch.tensor(
        np.stack([df['think_gain_a'].to_numpy(), df['think_gain_b'].to_numpy()], axis=1)
        if sample_count else np.zeros((0, 2)),
        dtype=torch.float32,
    )
    think_gain_mask = torch.tensor(
        np.stack([df['think_gain_mask_a'].to_numpy(), df['think_gain_mask_b'].to_numpy()], axis=1)
        if sample_count else np.zeros((0, 2)),
        dtype=torch.float32,
    )
    arrivals = torch.from_numpy(_decode_fixed_width_binary_column(
        df['arrivals'].to_list(), dtype=np.float32, width=NB_ARRIVAL_VALUES,
    ).copy())
    arrivals_mask = torch.tensor(df['arrivals_mask'].to_numpy(), dtype=torch.float32)
    return think_gain, think_gain_mask, arrivals, arrivals_mask


class RLDataset(torch.utils.data.Dataset):
    """
    Dataset for RL/self-play data with dual policy targets.
    """
    def __init__(self, x, y_value, policy_a, policy_b, wdl, moves_left, augment_flip=False):
        """
        Args:
            x: Input planes (N, 74, 8, 8)
            y_value: Value targets (N,)
            policy_a: Policy distribution for Board A (N, 4672)
            policy_b: Policy distribution for Board B (N, 4672)
            augment_flip: If True, doubles dataset size by including flipped samples
        """
        self.x = x
        self.y_value = y_value
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.wdl = wdl
        self.moves_left = moves_left
        self.augment_flip = augment_flip
        
    def __len__(self):
        # If augmentation is enabled, double the dataset size
        return len(self.x) * (2 if self.augment_flip else 1)
    
    def __getitem__(self, idx):
        # Determine if this is an original or flipped sample
        if self.augment_flip and idx >= len(self.x):
            # This is a flipped sample
            original_idx = idx - len(self.x)
            x, policy_a, policy_b = flip_bughouse_sample(
                self.x[original_idx], 
                self.policy_a[original_idx], 
                self.policy_b[original_idx]
            )
            return (
                x,
                self.y_value[original_idx],
                policy_a,
                policy_b,
                self.wdl[original_idx],
                self.moves_left[original_idx],
            )
        else:
            return (
                self.x[idx],
                self.y_value[idx],
                self.policy_a[idx],
                self.policy_b[idx],
                self.wdl[idx],
                self.moves_left[idx],
            )


class StreamingRLDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset for RL/self-play data that loads parquet files on-demand.
    
    This avoids loading all data into memory at once by streaming through
    parquet files and yielding samples one at a time.
    """
    def __init__(
        self,
        parquet_files: list,
        shuffle_files: bool = True,
        shuffle_buffer_size: int = 10000,
        augment_flip: bool = False,
        shuffle_samples: bool = True,
        include_joint_policy: bool = False,
        include_aux_labels: bool = False,
        include_sample_weights: bool = False,
    ):
        """
        Args:
            parquet_files: List of paths to parquet files
            shuffle_files: Whether to shuffle file order each epoch
            shuffle_buffer_size: Size of buffer for sample shuffling within files
            augment_flip: If True, yields both original and flipped versions of each sample
            shuffle_samples: If False, yield samples directly without retaining a
                shuffle buffer. This is useful for validation, where order does
                not matter and dense policy targets consume substantial RAM.
        """
        self.parquet_files = parquet_files
        self.shuffle_files = shuffle_files
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augment_flip = augment_flip
        self.shuffle_samples = shuffle_samples
        self.include_joint_policy = include_joint_policy
        self.include_aux_labels = include_aux_labels
        self.include_sample_weights = include_sample_weights
        
    def __iter__(self):
        files = self.parquet_files.copy()
        if self.shuffle_files:
            import random
            random.shuffle(files)

        def iter_shard(pf):
            """Keep decoded shard tensors scoped to one exhausted generator."""
            buffer = []
            if self.include_sample_weights:
                shard = load_rl_parquet_shard(
                    pf,
                    include_joint_policy=self.include_joint_policy,
                    include_aux_labels=self.include_aux_labels,
                    include_sample_weights=True,
                )
            elif self.include_aux_labels:
                shard = load_rl_parquet_shard(
                    pf,
                    include_joint_policy=self.include_joint_policy,
                    include_aux_labels=True,
                )
            elif self.include_joint_policy:
                shard = load_rl_parquet_shard(pf, include_joint_policy=True)
            else:
                shard = load_rl_parquet_shard(pf)
            x, y_val, pol_a, pol_b, wdl, moves_left = shard[:6]
            extras = shard[6:]
            weights = ()
            if self.include_sample_weights:
                weights = extras[-1:]
                extras = extras[:-1]
            aux = ()
            if self.include_aux_labels:
                aux = extras[-4:]
                extras = extras[:-4]
            joint = extras
            
            # Add samples to buffer (both original and flipped if augmentation is enabled)
            for i in range(len(x)):
                sample = (
                    x[i], y_val[i], pol_a[i], pol_b[i], wdl[i], moves_left[i]
                ) + tuple(values[i] for values in joint) + tuple(
                    values[i] for values in aux
                ) + tuple(values[i] for values in weights)
                if self.shuffle_samples:
                    buffer.append(sample)
                else:
                    yield sample
                
                # Add flipped sample if augmentation is enabled
                if self.augment_flip:
                    flipped_x, flipped_pol_a, flipped_pol_b = flip_bughouse_sample(
                        x[i], pol_a[i], pol_b[i]
                    )
                    flipped_sample = (
                        flipped_x,
                        y_val[i],
                        flipped_pol_a,
                        flipped_pol_b,
                        wdl[i],
                        moves_left[i],
                    )
                    if joint:
                        flipped_sample += (
                            joint[1][i],
                            joint[0][i],
                            joint[2][i],
                            joint[3][i],
                        )
                    if aux:
                        flipped_sample += flip_aux_labels(
                            *(values[i] for values in aux)
                        )
                    if weights:
                        flipped_sample += (weights[0][i],)
                    if self.shuffle_samples:
                        buffer.append(flipped_sample)
                    else:
                        yield flipped_sample
                
                # When buffer is full, shuffle and yield half
                if (
                    self.shuffle_samples
                    and len(buffer) >= self.shuffle_buffer_size
                ):
                    import random
                    random.shuffle(buffer)
                    # Yield first half
                    for sample in buffer[:len(buffer)//2]:
                        yield sample
                    buffer = buffer[len(buffer)//2:]

            if buffer:
                import random
                random.shuffle(buffer)
                yield from buffer

        # Samples are tensor views backed by the complete decoded shard. The
        # nested generator must be exhausted before the next file is decoded;
        # otherwise its locals can pin every dense tensor from the prior shard.
        for pf in files:
            yield from iter_shard(pf)


class CombinedRLDataset(torch.utils.data.IterableDataset):
    """
    Combines a regular RLDataset with a StreamingRLDataset.
    
    Yields all samples from the regular dataset first (shuffled),
    then streams from the streaming dataset.
    """
    def __init__(self, regular_dataset: RLDataset, streaming_dataset: StreamingRLDataset):
        self.regular_dataset = regular_dataset
        self.streaming_dataset = streaming_dataset
        
    def __iter__(self):
        import random
        
        # First yield from regular dataset (shuffled)
        indices = list(range(len(self.regular_dataset)))
        random.shuffle(indices)
        
        for idx in indices:
            yield self.regular_dataset[idx]
        
        # Then stream from streaming dataset
        for sample in self.streaming_dataset:
            yield sample


def load_rl_data_from_directory(data_dir, max_samples=None):
    """
    Load all RL parquet shards from a directory.
    
    Args:
        data_dir: Directory containing RL parquet files
        max_samples: Optional limit on total samples to load
        
    Returns:
        Tuple of (x, y_value, policy_a, policy_b, wdl, moves_left) tensors
    """
    import glob
    from pathlib import Path
    
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    all_x = []
    all_y_value = []
    all_policy_a = []
    all_policy_b = []
    all_wdl = []
    all_moves_left = []
    
    total_loaded = 0
    for pf in parquet_files:
        x, y_val, pol_a, pol_b, wdl, moves_left = load_rl_parquet_shard(pf)
        all_x.append(x)
        all_y_value.append(y_val)
        all_policy_a.append(pol_a)
        all_policy_b.append(pol_b)
        all_wdl.append(wdl)
        all_moves_left.append(moves_left)
        
        total_loaded += len(x)
        if max_samples and total_loaded >= max_samples:
            break
    
    x_tensor = torch.cat(all_x, dim=0)
    y_val_tensor = torch.cat(all_y_value, dim=0)
    policy_a_tensor = torch.cat(all_policy_a, dim=0)
    policy_b_tensor = torch.cat(all_policy_b, dim=0)
    wdl_tensor = torch.cat(all_wdl, dim=0)
    moves_left_tensor = torch.cat(all_moves_left, dim=0)
    
    if max_samples:
        x_tensor = x_tensor[:max_samples]
        y_val_tensor = y_val_tensor[:max_samples]
        policy_a_tensor = policy_a_tensor[:max_samples]
        policy_b_tensor = policy_b_tensor[:max_samples]
        wdl_tensor = wdl_tensor[:max_samples]
        moves_left_tensor = moves_left_tensor[:max_samples]
    
    return (
        x_tensor,
        y_val_tensor,
        policy_a_tensor,
        policy_b_tensor,
        wdl_tensor,
        moves_left_tensor,
    )
