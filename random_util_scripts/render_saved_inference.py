#!/usr/bin/env python3
"""Render saved inference PNGs into frame-numbered mp4 videos.

Combines the two former steps (make_inference_videos.sh + add_frame_numbers.py)
into a single script, and adds glob-pattern selection of session folders.

For every folder matching the input glob pattern, this looks for
    <session>/deployment_time_inference/images
    <session>/policy_time_inference/images
renders each set of PNGs into an mp4 (with the frame index burned into the
top-left corner in a single pass), and writes them to:
    <output_dir>/deployment_videos/<session>_<deploy_fps>Hz.mp4
    <output_dir>/policy_videos/<session>_<policy_fps>Hz.mp4

Example:
    python random_util_scripts/render_saved_inference.py \
        "../saved_inference/2026-06-25*"
"""

import argparse
import glob
import os
import re
from pathlib import Path

import cv2

# Filenames look like:
#   deployment: img_{N}_batch0_img0.png
#   policy:     img_{s}_{counter}_batch0_img0.png
#               img_initial_{counter}_batch0_img0.png
DEPLOY_RE = re.compile(r"^img_[0-9]+_batch0_img0\.png$")
POLICY_RE = re.compile(r"^img_.*_batch0_img0\.png$")


def list_deploy_images(images_dir: Path) -> list[Path]:
    """Sorted deployment PNGs, ordered numerically by the integer in field 2."""
    names = [n for n in os.listdir(images_dir) if DEPLOY_RE.match(n)]
    # img_{N}_batch0_img0.png -> N is split('_')[1]
    names.sort(key=lambda n: int(n.split("_")[1]))
    return [images_dir / n for n in names]


def list_policy_images(images_dir: Path) -> list[Path]:
    """Sorted policy PNGs, ordered numerically by the $(NF-2) underscore field."""
    names = [n for n in os.listdir(images_dir) if POLICY_RE.match(n)]
    # tokens: [..., counter, "batch0", "img0.png"] -> counter is tokens[-3]
    names.sort(key=lambda n: int(n.split("_")[-3]))
    return [images_dir / n for n in names]


def render_images_to_video(image_paths: list[Path], output_path: Path, fps: float) -> None:
    """Write image_paths to an mp4 at fps, burning the frame index top-left."""
    if not image_paths:
        print(f"    SKIP: no images to render for {output_path}")
        return

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    margin = 10

    written = 0
    for frame_idx, path in enumerate(image_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"    WARNING: could not read {path}, skipping")
            continue

        text = str(frame_idx)
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = margin, margin + text_h

        # Dark background rect for readability
        cv2.rectangle(frame, (x - 4, y - text_h - 4), (x + text_w + 4, y + baseline + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        out.write(frame)
        written += 1

    out.release()
    print(f"    saved: {output_path} ({written} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", help='Glob pattern matching session folders, e.g. "../saved_inference/2026-06-25*"')
    parser.add_argument("--output-dir", default=None,
                        help="Where to create deployment_videos/ and policy_videos/ (default: common parent of matched folders)")
    parser.add_argument("--deploy-fps", type=float, default=60.0, help="Deployment video frame rate (default: 60)")
    parser.add_argument("--policy-fps", type=float, default=10.0, help="Policy video frame rate (default: 10)")
    parser.add_argument("--overwrite", action="store_true", help="Re-render even if the output mp4 already exists")
    args = parser.parse_args()

    matches = sorted(p for p in glob.glob(args.pattern) if os.path.isdir(p))
    if not matches:
        parser.error(f"No directories matched pattern: {args.pattern}")

    if args.output_dir is not None:
        output_root = Path(args.output_dir)
    else:
        output_root = Path(os.path.commonpath([os.path.dirname(os.path.abspath(m)) for m in matches]))

    deploy_out = output_root / "deployment_videos"
    policy_out = output_root / "policy_videos"
    deploy_out.mkdir(parents=True, exist_ok=True)
    policy_out.mkdir(parents=True, exist_ok=True)

    print(f"Matched {len(matches)} folder(s); output root: {output_root}")

    fps_label = lambda fps: str(int(fps)) if float(fps).is_integer() else str(fps)

    for session in matches:
        session = session.rstrip("/")
        datestr = os.path.basename(session)
        print(f"\nSession: {session}")

        jobs = [
            ("deployment", Path(session) / "deployment_time_inference" / "images",
             deploy_out / f"{datestr}_{fps_label(args.deploy_fps)}Hz.mp4", args.deploy_fps, list_deploy_images),
            ("policy", Path(session) / "policy_time_inference" / "images",
             policy_out / f"{datestr}_{fps_label(args.policy_fps)}Hz.mp4", args.policy_fps, list_policy_images),
        ]

        for label, images_dir, out_mp4, fps, lister in jobs:
            if out_mp4.exists() and not args.overwrite:
                print(f"  SKIP (exists): {out_mp4.name}")
                continue
            if not images_dir.is_dir():
                print(f"  SKIP: {images_dir} not found")
                continue
            print(f"  ==> Building {label} video ({fps_label(fps)} Hz)...")
            render_images_to_video(lister(images_dir), out_mp4, fps)

    print("\nDone.")


if __name__ == "__main__":
    main()


"""

python3 -u random_util_scripts/render_saved_inference.py \
"../saved_inference/2026-06-25*" \
--output-dir ../saved_inference \
--deploy-fps 120 \
--policy-fps 60 


"""