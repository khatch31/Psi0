#!/usr/bin/env python3
"""Add frame numbers to the top-left corner of each frame in an mp4 video."""

import argparse
import cv2


def add_frame_numbers(input_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        text = str(frame_idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        margin = 10

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = margin, margin + text_h

        # Dark background rect for readability
        cv2.rectangle(frame, (x - 4, y - text_h - 4), (x + text_w + 4, y + baseline + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Wrote {frame_idx} frames to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay frame numbers on an mp4 video.")
    parser.add_argument("input", help="Path to input mp4 file")
    parser.add_argument("output", help="Path to output mp4 file")
    args = parser.parse_args()

    add_frame_numbers(args.input, args.output)


"""

python3 -u random_util_scripts/add_frame_numbers.py \
../saved_inference/2026-05-15_15-57-17/deployment_video_30hz.mp4 \
../saved_inference/2026-05-15_15-57-17/deployment_video_30hz_timesteps.mp4


"""