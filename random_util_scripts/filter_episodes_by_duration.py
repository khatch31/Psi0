"""
Filter a LeRobot-format dataset to only keep episodes whose video is <= MAX_DURATION seconds.
Episode numbers are preserved as-is (no renumbering).

Usage:
    python filter_episodes_by_duration.py
"""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

SRC = Path("/home/ubuntu/Desktop/world_models_project/psi0_workspace/data/simple/G1WholebodyHandoverTeleop-v0")
DST = Path("/home/ubuntu/Desktop/world_models_project/psi0_workspace/data/simple/G1WholebodyHandoverTeleop-v0-filtered")
MAX_DURATION = 8.0
VIDEO_SUBDIR = "egocentric"
MANUAL_SKIP_EPISODES = [7, 88]


def get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    video_dir = SRC / "videos" / "chunk-000" / VIDEO_SUBDIR
    keep = {int(f.stem.split("_")[1]) for f in video_dir.glob("episode_*.mp4")
            if get_duration(f) <= MAX_DURATION}
    
    keep = {ep_no for ep_no in keep if ep_no not in MANUAL_SKIP_EPISODES}

    

    print(f"Keeping {len(keep)} / {len(list(video_dir.glob('episode_*.mp4')))} episodes")

    # Create output dirs
    (DST / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (DST / "videos" / "chunk-000" / VIDEO_SUBDIR).mkdir(parents=True, exist_ok=True)
    (DST / "meta").mkdir(parents=True, exist_ok=True)

    # ### CLAUDE ### Renumber episodes to contiguous 0-based indices
    keep_sorted = sorted(keep)
    old_to_new = {old: new for new, old in enumerate(keep_sorted)}

    # Copy/rewrite parquet files with updated episode_index and index columns;
    # copy video files renaming to new contiguous indices
    global_frame_idx = 0
    for old_idx, new_idx in sorted(old_to_new.items(), key=lambda kv: kv[1]):
        src_parquet = SRC / "data" / "chunk-000" / f"episode_{old_idx:06d}.parquet"
        dst_parquet = DST / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"
        df = pd.read_parquet(src_parquet)
        df["episode_index"] = new_idx
        df["index"] = range(global_frame_idx, global_frame_idx + len(df))
        df.to_parquet(dst_parquet, index=False)
        global_frame_idx += len(df)

        shutil.copy2(SRC / "videos" / "chunk-000" / VIDEO_SUBDIR / f"episode_{old_idx:06d}.mp4",
                     DST / "videos" / "chunk-000" / VIDEO_SUBDIR / f"episode_{new_idx:06d}.mp4")

    # Load all kept entries from source, keyed by old episode_index
    src_episodes = {}
    src_episodes_stats = {}
    with open(SRC / "meta" / "episodes.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            if entry["episode_index"] in keep:
                src_episodes[entry["episode_index"]] = entry
    with open(SRC / "meta" / "episodes_stats.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            if entry["episode_index"] in keep:
                src_episodes_stats[entry["episode_index"]] = entry

    # Write episodes.jsonl with renumbered indices and recomputed frame offsets
    frame_offset = 0
    with open(DST / "meta" / "episodes.jsonl", "w") as dst_f:
        for old_idx in keep_sorted:
            entry = dict(src_episodes[old_idx])
            length = entry["length"]
            entry["episode_index"] = old_to_new[old_idx]
            entry["dataset_from_index"] = frame_offset
            entry["dataset_to_index"] = frame_offset + length
            frame_offset += length
            dst_f.write(json.dumps(entry) + "\n")

    # Write episodes_stats.jsonl with renumbered indices
    with open(DST / "meta" / "episodes_stats.jsonl", "w") as dst_f:
        for old_idx in keep_sorted:
            entry = dict(src_episodes_stats[old_idx])
            entry["episode_index"] = old_to_new[old_idx]
            dst_f.write(json.dumps(entry) + "\n")
    # ### END CLAUDE ###

    # Update info.json totals
    with open(SRC / "meta" / "info.json") as f:
        info = json.load(f)
    total_frames = frame_offset
    info["total_episodes"] = len(keep_sorted)
    info["total_frames"] = total_frames
    info["total_videos"] = len(keep_sorted)
    with open(DST / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Copy unchanged meta files
    for fname in ("tasks.jsonl", "modality.json", "lang_map.json",
                  "stats.json", "stats_psi0.json", "relative_stats.json"):
        if (SRC / "meta" / fname).exists():
            shutil.copy2(SRC / "meta" / fname, DST / "meta" / fname)

    print(f"Done -> {DST}  ({len(keep_sorted)} episodes, {total_frames} frames)")


if __name__ == "__main__":
    main()
