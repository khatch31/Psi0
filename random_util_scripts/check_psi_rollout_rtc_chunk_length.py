import os
import sys
import numpy as np
import json
import yaml
import pickle
from collections import defaultdict
from glob import glob
from tqdm import tqdm, trange
import pandas as pd
import cv2
import imageio
import copy
import shutil

from rich.console import Console
console = Console()
def color_print(*args, markup=False, style='red'):
    console.print(*args, style=style, markup=markup)

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_yaml(file_path):
    with open(file_path) as f:
        return yaml.safe_load(f)
    
def load_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data

def write_pickle(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def write_yaml(file_path, data, default_flow_style=False, sort_keys=False):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys)

def write_json(filepath, data, indent=2):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)

PSI_HOME = os.environ["PSI_HOME"]

if __name__ == "__main__":
    INF_DIR = f"{PSI_HOME}/saved_inference/2026-06-19/2026-06-19_12-50-05"

    timestep_no_from_file = lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
    action_info_files = glob(os.path.join(INF_DIR, "deployment_time_inference", "action_infos", "action_infos_*.npy"))
    action_info_files = sorted(action_info_files, key=timestep_no_from_file)

    pred_action_files = glob(os.path.join(INF_DIR, "deployment_time_inference", "pred_actions", "pred_actions_*.npy"))
    pred_action_files = sorted(pred_action_files, key=timestep_no_from_file)

    obs_files = glob(os.path.join(INF_DIR, "deployment_time_inference", "observations", "obs_*.pkl"))
    obs_files = sorted(obs_files, key=timestep_no_from_file)

    assert len(action_info_files) == len(obs_files) == len(pred_action_files)
    assert timestep_no_from_file(action_info_files[0]) == timestep_no_from_file(obs_files[0]) == timestep_no_from_file(pred_action_files[0]) == 0
    assert timestep_no_from_file(action_info_files[-1]) == timestep_no_from_file(obs_files[-1]) == timestep_no_from_file(pred_action_files[-1]) == len(action_info_files) - 1

    prediction_idxs = []
    timestamps = []
    pred_actions = []

    for t in trange(len(action_info_files)):
        action_info_file = action_info_files[t]
        pred_action_file = pred_action_files[t]
        obs_file = obs_files[t]
        action_info = np.load(action_info_file)
        pred_action = np.load(pred_action_file)
        # obs = load_pickle(obs_file)

        assert action_info.shape == (1, 2)
        action_info = np.squeeze(action_info)

        prediction_idx = action_info[0]
        timestamp = action_info[1]

        
        prediction_idxs.append(prediction_idx)
        timestamps.append(timestamp)
        pred_actions.append(pred_action)

        if t < 200:
            color_print(f"[{t}] prediction_idx: {prediction_idx}, timestamp: {timestamp}, np.linalg.norm(pred_action): {np.linalg.norm(pred_action)}", style="green")

    # Cast to int so run boundaries use exact integer equality rather than float comparison
    prediction_idxs = np.array(prediction_idxs).astype(int)

    # Find run lengths: the prediction index changes wherever consecutive values differ.
    # np.diff != 0 marks the boundaries between runs; the indices of those boundaries let us
    # split the sequence into runs of constant prediction index.
    change_points = np.flatnonzero(np.diff(prediction_idxs) != 0) + 1
    run_lengths = np.diff([0, *change_points, len(prediction_idxs)])

    avg_run_length = run_lengths.mean()
    std_run_length = run_lengths.std()
    color_print(
        f"prediction_idx stayed constant for an average of {avg_run_length:.4f} +/- {std_run_length:.4f} timesteps "
        f"({len(run_lengths)} runs over {len(prediction_idxs)} timesteps, "
        f"min={run_lengths.min()}, max={run_lengths.max()})",
        style="green",
    )

    # Do the same in terms of actual elapsed time. Each timestep contributes the interval
    # (current timestamp - previous timestamp); the first timestep has no predecessor so it
    # contributes 0. Summing these per run gives how long (in real time) the prediction index
    # stayed constant. run_starts are the first-timestep indices of each run, and reduceat sums
    # the intervals between consecutive run starts.
    timestamps = np.array(timestamps)
    intervals = np.diff(timestamps, prepend=timestamps[0])
    run_starts = np.concatenate([[0], change_points])
    run_durations = np.add.reduceat(intervals, run_starts)

    avg_run_duration = run_durations.mean()
    std_run_duration = run_durations.std()
    color_print(
        f"prediction_idx stayed constant for an average of {avg_run_duration:.4f} +/- {std_run_duration:.4f} seconds "
        f"({len(run_durations)} runs over {timestamps[-1] - timestamps[0]:.4f} total seconds, "
        f"min={run_durations.min():.4f}, max={run_durations.max():.4f})",
        style="green",
    )


    import ipdb; ipdb.set_trace()