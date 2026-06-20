"""
driving_command_based_traj.py

Unified pipeline for generating driving-command-based vocab cluster labels (3 steps).

Each vocab entry can belong to multiple command categories simultaneously
(multi-hot assignment). e.g. a slightly-curving-left trajectory may be
labeled as both 'straight' and 'left'.

  Step 1: For each vocab entry, set a flag for every command that received
          at least one vote from GT log matching.
  Step 2: Fill unmatched entries (no flags set) by copying flags from the
          nearest matched entry.
  Step 3: Fix left/right flag mismatches based on trajectory endpoint direction.

The original vocab (N, 40, 3) is left untouched.

Internal representation:
  cmd_flags (N, 3) bool — columns: [straight, right, left]

Output pkl structure (indices may overlap across commands):
  {
    'straight': np.ndarray  — vocab indices with straight flag set
    'right':    np.ndarray  — vocab indices with right flag set
    'left':     np.ndarray  — vocab indices with left flag set
  }

Usage in model:
  for k, v in self.vocab_cluster.items():
      self.register_buffer(f"vocab_cluster_{k}", torch.tensor(v, dtype=torch.long))

Example:
  python driving_command_based_traj.py \\
      --vocab_path traj_final/16384.npy \\
      --log_dirs /data/navsim_logs/trainval /data/navsim_logs/test \\
      --output traj_final/cluster_labels_16384.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
_DEFAULT_VOCAB  = "traj_final/16384.npy"
_DEFAULT_OUTPUT = "traj_final/cluster_labels_16384.pkl"

SAMPLED_TIMEPOINTS = [5 * k - 1 for k in range(1, 9)]   # [4,9,14,19,24,29,34,39]

NUM_HISTORY_FRAMES = 4
NUM_FUTURE_FRAMES  = 10
TOTAL_FRAMES       = NUM_HISTORY_FRAMES + NUM_FUTURE_FRAMES
NUM_GT_FRAMES      = 8

# nuPlan log driving_command order: [left, straight, right, unknown]
# cmd_flags column order:           [straight(0), right(1), left(2)]
# log index → cmd_flags column:  straight=log[1], right=log[2], left=log[0]
CMD_NAMES = ['straight', 'right', 'left']   # maps to cmd_flags columns 0,1,2


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _get_global_pose(frame: dict) -> np.ndarray:
    from pyquaternion import Quaternion
    t = frame["ego2global_translation"]
    q = Quaternion(*frame["ego2global_rotation"])
    return np.array([t[0], t[1], q.yaw_pitch_roll[0]], dtype=np.float64)


def _get_gt_local_traj(frame_list: list, history_end_idx: int, num_future: int) -> np.ndarray:
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )
    origin_pose   = _get_global_pose(frame_list[history_end_idx])
    future_global = np.array(
        [_get_global_pose(frame_list[history_end_idx + i + 1]) for i in range(num_future)],
        dtype=np.float64,
    )
    return convert_absolute_to_relative_se2_array(StateSE2(*origin_pose), future_global)


def _save_pkl(cmd_flags: np.ndarray, output_path: Path):
    """Save cmd_flags (N, 3) as {command_name: indices_array} pkl.
    Indices may overlap across commands (multi-hot).
    """
    result = {
        name: np.where(cmd_flags[:, col])[0].astype(np.int64)
        for col, name in enumerate(CMD_NAMES)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    N = cmd_flags.shape[0]
    print(f"[saved] {output_path}")
    for name, indices in result.items():
        print(f"  {name:10s}: {len(indices):,} entries  ({len(indices)/N*100:.1f}%)")
    n_multi = int((cmd_flags.sum(axis=1) > 1).sum())
    n_none  = int((cmd_flags.sum(axis=1) == 0).sum())
    print(f"  multi-label : {n_multi:,}  ({n_multi/N*100:.1f}%)")
    print(f"  no label    : {n_none:,}  ({n_none/N*100:.1f}%)")


# ──────────────────────────────────────────────────────────────
# Step 1: GT log → assign multi-hot command flags to each vocab entry
# ──────────────────────────────────────────────────────────────

def step1_assign_command(args, vocab: np.ndarray, output_path: Path) -> np.ndarray:
    """
    For each vocab entry, set a flag for every command that received >= 1 vote.
    Unknown votes are ignored.

    Returns:
        cmd_flags (N, 3) bool — [straight, right, left]
    """
    print("\n" + "=" * 60)
    print("Step 1: Assign multi-hot driving command flags from GT logs")
    print("=" * 60)

    N = vocab.shape[0]
    assert vocab.shape[1] == 40 and vocab.shape[2] == 3, \
        f"Invalid vocab shape: {vocab.shape}, expected (N,40,3)"
    print(f"[vocab] shape={vocab.shape}")

    vocab_sampled = vocab[:, SAMPLED_TIMEPOINTS, :]   # (N, 8, 3)

    # vote_counts (N, 4): nuPlan log order [left, straight, right, unknown]
    vote_counts = np.zeros((N, 4), dtype=np.int64)

    log_files = []
    for d in [Path(d) for d in args.log_dirs]:
        log_files.extend(sorted(d.glob("*.pkl")))
    print(f"[logs] search dirs: {[str(d) for d in args.log_dirs]}")
    print(f"[logs] log files to process: {len(log_files)}")

    total_scenes = matched_scenes = 0

    for log_path in tqdm(log_files, desc="Processing logs"):
        with open(log_path, "rb") as f:
            frame_list_all = pickle.load(f)

        n_frames = len(frame_list_all)
        for start in range(0, n_frames - TOTAL_FRAMES + 1):
            scene_frames = frame_list_all[start: start + TOTAL_FRAMES]
            history_end  = NUM_HISTORY_FRAMES - 1

            total_scenes += 1

            gt_poses = _get_gt_local_traj(scene_frames, history_end, NUM_GT_FRAMES)

            raw_cmd = scene_frames[history_end].get("driving_command", None)
            if raw_cmd is None:
                continue
            cmd_vec = np.asarray(raw_cmd, dtype=np.float32)
            if cmd_vec.shape[0] != 4:
                continue

            diff     = vocab_sampled - gt_poses[None]
            l2_sum   = (diff ** 2).sum(axis=(1, 2))
            best_idx = int(np.argmin(l2_sum))

            # nuPlan log order: [left(0), straight(1), right(2), unknown(3)]
            log_class = int(np.argmax(cmd_vec))
            vote_counts[best_idx, log_class] += 1
            matched_scenes += 1

    print(f"\n[result] total scenes={total_scenes:,}, matched scenes={matched_scenes:,}")

    # multi-hot: flag is set if that command received >= 1 vote (unknown ignored)
    # nuPlan log cols: left=0, straight=1, right=2  →  cmd_flags: straight=0, right=1, left=2
    cmd_flags = np.zeros((N, 3), dtype=bool)
    cmd_flags[:, 0] = vote_counts[:, 1] > 0   # straight
    cmd_flags[:, 1] = vote_counts[:, 2] > 0   # right
    cmd_flags[:, 2] = vote_counts[:, 0] > 0   # left

    matched_mask = cmd_flags.any(axis=1)
    print(f"[matched] vocab entries with >= 1 vote: {matched_mask.sum():,} / {N:,}")

    _save_pkl(cmd_flags, output_path)
    return cmd_flags


# ──────────────────────────────────────────────────────────────
# Step 2: Fill unmatched entries with nearest matched entry's flags
# ──────────────────────────────────────────────────────────────

def step2_fill_none(vocab: np.ndarray, cmd_flags: np.ndarray,
                    output_path: Path, batch_size: int = 64) -> np.ndarray:
    """
    Args:
        vocab:     (N, 40, 3)
        cmd_flags: (N, 3) bool — [straight, right, left]
    Returns:
        cmd_flags: (N, 3) with unmatched entries filled
    """
    print("\n" + "=" * 60)
    print("Step 2: Fill unmatched entries with nearest matched flags")
    print("=" * 60)

    none_mask    = ~cmd_flags.any(axis=1)    # entries with no flag set
    matched_mask = ~none_mask

    n_none    = int(none_mask.sum())
    n_matched = int(matched_mask.sum())
    print(f"[stats] unmatched entries: {n_none:,}")
    print(f"[stats] matched entries  : {n_matched:,}")

    if n_none == 0:
        print("[done] No unmatched entries — saving without changes.")
        _save_pkl(cmd_flags, output_path)
        return cmd_flags

    vocab_sampled = vocab[:, SAMPLED_TIMEPOINTS, :]    # (N, 8, 3)
    none_traj     = vocab_sampled[none_mask]            # (M, 8, 3)
    matched_traj  = vocab_sampled[matched_mask]         # (K, 8, 3)
    matched_flags = cmd_flags[matched_mask]             # (K, 3)

    print("[processing] Computing L2 nearest-neighbor...")
    M = none_traj.shape[0]
    nearest_flags = np.zeros((M, 3), dtype=bool)

    for start in tqdm(range(0, M, batch_size), desc="nearest-neighbor"):
        end   = min(start + batch_size, M)
        batch = none_traj[start:end]                           # (b, 8, 3)
        diff  = batch[:, None, :, :] - matched_traj[None, :, :, :]
        l2    = (diff ** 2).sum(axis=(2, 3))
        best  = np.argmin(l2, axis=1)
        nearest_flags[start:end] = matched_flags[best]

    flags_new = cmd_flags.copy()
    flags_new[none_mask] = nearest_flags

    still_none = int((~flags_new.any(axis=1)).sum())
    print(f"[verify] remaining unmatched entries: {still_none}")

    _save_pkl(flags_new, output_path)
    return flags_new


# ──────────────────────────────────────────────────────────────
# Step 3: Fix left/right flag mismatches
# ──────────────────────────────────────────────────────────────

def step3_fix_mismatch(vocab: np.ndarray, cmd_flags: np.ndarray,
                       output_path: Path) -> np.ndarray:
    """
    ego frame: y > 0 = LEFT, y < 0 = RIGHT

    Fix rules (applied to flags, not dominant command):
      LEFT  traj (endpoint y > 0) + right flag set → clear right, set left
      RIGHT traj (endpoint y < 0) + left  flag set → clear left,  set right

    Args:
        vocab:     (N, 40, 3)
        cmd_flags: (N, 3) bool — [straight, right, left]
    Returns:
        cmd_flags: (N, 3) corrected
    """
    print("\n" + "=" * 60)
    print("Step 3: Fix left/right flag mismatches")
    print("=" * 60)

    endpoint_y = vocab[:, -1, 1]   # (N,)
    left_traj  = endpoint_y > 0    # ego y > 0 = left turn
    right_traj = endpoint_y < 0    # ego y < 0 = right turn

    print(f"[stats] LEFT  traj (y > 0): {int(left_traj.sum()):,}")
    print(f"[stats] RIGHT traj (y < 0): {int(right_traj.sum()):,}")

    # col 1 = right flag, col 2 = left flag
    wrong_right = cmd_flags[:, 1] & left_traj    # left traj but right flag set
    wrong_left  = cmd_flags[:, 2] & right_traj   # right traj but left flag set

    print(f"\n[before fix] LEFT  traj + right flag: {int(wrong_right.sum()):,}")
    print(f"[before fix] RIGHT traj + left  flag: {int(wrong_left.sum()):,}")

    flags_new = cmd_flags.copy()
    flags_new[wrong_right, 1] = False   # clear right
    flags_new[wrong_right, 2] = True    # set left
    flags_new[wrong_left,  2] = False   # clear left
    flags_new[wrong_left,  1] = True    # set right

    print(f"[fixed] right→left: {int(wrong_right.sum()):,} entries")
    print(f"[fixed] left→right: {int(wrong_left.sum()):,} entries")
    print(f"\n[after fix] LEFT  traj + right flag: {int((flags_new[:, 1] & left_traj).sum()):,}")
    print(f"[after fix] RIGHT traj + left  flag: {int((flags_new[:, 2] & right_traj).sum()):,}")

    _save_pkl(flags_new, output_path)
    return flags_new


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-hot driving-command vocab cluster labels (Step 1→2→3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python driving_command_based_traj.py \\
      --vocab_path traj_final/16384.npy \\
      --log_dirs /data/navsim_logs/trainval /data/navsim_logs/test \\
      --output traj_final/cluster_labels_16384.pkl
        """,
    )

    parser.add_argument("--vocab_path", default=_DEFAULT_VOCAB,
                        help=f"Input vocab .npy (N,40,3)  (default: {_DEFAULT_VOCAB})")
    parser.add_argument("--log_dirs",   nargs="+", required=True,
                        help="Log directories containing .pkl files")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for Step 2 nearest-neighbor (default: 64)")
    parser.add_argument("--output",     default=_DEFAULT_OUTPUT,
                        help=f"Final output .pkl  (default: {_DEFAULT_OUTPUT})")

    args = parser.parse_args()

    out     = Path(args.output)
    s1_path = out.parent / f"{out.stem}_step1.pkl"
    s2_path = out.parent / f"{out.stem}_step2.pkl"
    s3_path = out

    vocab = np.load(args.vocab_path)
    print(f"[vocab] loaded: {args.vocab_path}  shape={vocab.shape}")

    cmd_flags = step1_assign_command(args, vocab, s1_path)
    cmd_flags = step2_fill_none(vocab, cmd_flags, s2_path, args.batch_size)
    cmd_flags = step3_fix_mismatch(vocab, cmd_flags, s3_path)

    print("\n" + "=" * 60)
    print(f"Done  →  {s3_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
