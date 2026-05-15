"""Dummy client for testing serve_psi0-rtc.sh.

Loads the same dataset used for training, pulls the first frame, and
repeatedly sends that single observation to the policy server over
WebSocket at ~30Hz. Received actions are printed.

Usage:


source .env
source .venv-psi/bin/activate
python3 -u scripts/deploy/dummy_client_psi0-rtc.py \
--run-dir /home/songlin/Projects/psi0_kyle/psi0_workspace/training_output/finetune/open_a_drawer_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2605062254 \
--host localhost --port 8014


--run-dir /home/ubuntu/Desktop/world_models_project/psi0_workspace/training_output/finetune/open_a_drawer_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2605062254 \

"""
import argparse
import json
import os
import sys
import threading
import time
from base64 import b64decode, b64encode
from pathlib import Path

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from websocket import WebSocketApp

DEFAULT_PSI_HOME = os.environ["PSI_HOME"]

DEFAULT_RUN_DIR = (
    f"{DEFAULT_PSI_HOME}/training_output/"
    "finetune/open_a_drawer_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2605062254"
)
# DEFAULT_PSI_HOME = "/home/ubuntu/Desktop/world_models_project/psi0_workspace"
# DEFAULT_PSI_HOME = "/home/songlin/Projects/psi0_kyle/psi0_workspace"

OBS_SEND_HZ = 30
OBS_SEND_INTERVAL = 1.0 / OBS_SEND_HZ


# ----------------------------------------------------------------------------
# Numpy <-> JSON helpers (must match src/psi/deploy/helpers.py)
# ----------------------------------------------------------------------------
def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        arr = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return arr.reshape(dct["shape"]) if dct["shape"] else arr[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {k: convert_numpy_in_dict(v, func) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy_in_dict(x, func) for x in data]
    if isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


# ----------------------------------------------------------------------------
# Build first observation from the LeRobot dataset (mirrors openloop_eval.ipynb)
# ----------------------------------------------------------------------------
def build_first_observation(run_dir: Path):
    # os.environ.setdefault("PSI_HOME", DEFAULT_PSI_HOME)

    # Lazy-import to avoid pulling psi deps unless needed
    from psi.config.config import LaunchConfig
    from psi.data.lerobot.lerobot_ext import LeRobotDatasetWrapper
    from psi.utils import parse_args_to_tyro_config, seed_everything

    config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
    conf = (run_dir / "run_config.json").open("r").read()
    launch_config = config_.model_validate_json(conf)
    seed_everything(launch_config.seed or 42)

    data_cfg = launch_config.data
    image_key = data_cfg.transform.repack.image_keys[0]

    # Skip the full transform pipeline (which would require a VLM processor);
    # grab the raw LeRobot frame directly.
    wrapper = LeRobotDatasetWrapper(data_cfg, split="train")
    frame = wrapper[0]

    img = frame[image_key]
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:  # (T, C, H, W) -> take first frame
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    states = frame["states"]
    if hasattr(states, "cpu"):
        states = states.cpu().numpy()
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 2:  # (To, Ds) -> take latest
        states = states[-1]
    # State layout matches server's _parse_obs_payload: [hand(14), arm(14), ...]
    hand_joints = states[:14].astype(np.float32)
    arm_joints = states[14:28].astype(np.float32)

    instruction = frame.get("task", "")
    if not isinstance(instruction, str):
        instruction = str(instruction)

    print(f"[dummy] image_key={image_key} image_shape={img.shape} dtype={img.dtype}")
    print(f"[dummy] hand_joints={hand_joints.shape} arm_joints={arm_joints.shape}")
    print(f"[dummy] instruction='{instruction}'")

    return {
        "image": {image_key: img},
        "state": {"hand_joints": hand_joints, "arm_joints": arm_joints},
        "instruction": instruction,
        "history": None,
        "condition": None,
        "gt_action": None,
        "dataset_name": None,
        "timestamp": None,
    }


### CLAUDE ### Load every observation from the first episode so the client can cycle through them
def _frame_to_payload(frame, image_key):
    """Convert one LeRobot frame to a server payload dict (matches build_first_observation)."""
    img = frame[image_key]
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:  # (T, C, H, W) -> take first frame
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    states = frame["states"]
    if hasattr(states, "cpu"):
        states = states.cpu().numpy()
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 2:  # (To, Ds) -> take latest
        states = states[-1]
    hand_joints = states[:14].astype(np.float32)
    arm_joints = states[14:28].astype(np.float32)

    instruction = frame.get("task", "")
    if not isinstance(instruction, str):
        instruction = str(instruction)

    return {
        "image": {image_key: img},
        "state": {"hand_joints": hand_joints, "arm_joints": arm_joints},
        "instruction": instruction,
        "history": None,
        "condition": None,
        "gt_action": None,
        "dataset_name": None,
        "timestamp": None,
    }


def _get_episode_index(frame):
    """Best-effort extraction of episode_index from a LeRobot frame; None if absent."""
    if not isinstance(frame, dict):
        return None
    ep = frame.get("episode_index")
    if ep is None:
        return None
    if hasattr(ep, "item"):
        try:
            return int(ep.item())
        except Exception:
            pass
    if isinstance(ep, np.ndarray):
        return int(ep.flat[0])
    try:
        return int(ep)
    except Exception:
        return None


def build_first_episode_observations(run_dir: Path):
    """Load every frame from the first episode of the training dataset."""
    from psi.config.config import LaunchConfig
    from psi.data.lerobot.lerobot_ext import LeRobotDatasetWrapper
    from psi.utils import parse_args_to_tyro_config, seed_everything

    config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
    conf = (run_dir / "run_config.json").open("r").read()
    launch_config = config_.model_validate_json(conf)
    seed_everything(launch_config.seed or 42)

    data_cfg = launch_config.data
    image_key = data_cfg.transform.repack.image_keys[0]

    wrapper = LeRobotDatasetWrapper(data_cfg, split="train")

    payloads = []
    first_ep = None
    for idx in range(len(wrapper)):
        frame = wrapper[idx]
        ep = _get_episode_index(frame)
        if first_ep is None:
            first_ep = ep
        elif ep is not None and ep != first_ep:
            break
        payloads.append(_frame_to_payload(frame, image_key))

    if not payloads:
        raise RuntimeError("No frames loaded from dataset")

    sample_img = payloads[0]["image"][image_key]
    sample_hand = payloads[0]["state"]["hand_joints"]
    sample_arm = payloads[0]["state"]["arm_joints"]
    print(
        f"[dummy] image_key={image_key} image_shape={sample_img.shape} dtype={sample_img.dtype}"
    )
    print(f"[dummy] hand_joints={sample_hand.shape} arm_joints={sample_arm.shape}")
    print(f"[dummy] instruction='{payloads[0]['instruction']}'")
    print(f"[dummy] loaded {len(payloads)} observations from episode {first_ep}")
    return payloads
### END CLAUDE ###


# ----------------------------------------------------------------------------
# WebSocket client
# ----------------------------------------------------------------------------
class DummyRTCClient:
    def __init__(self, server_url: str, payloads, max_steps: int | None = None):
        self.server_url = server_url
        ### CLAUDE ### Accept a list of payloads (one per timestep) and pre-serialize each.
        # self.payload_json = json.dumps(convert_numpy_in_dict(payload, numpy_serialize))
        if isinstance(payloads, dict):
            payloads = [payloads]
        self.payload_jsons = [
            json.dumps(convert_numpy_in_dict(p, numpy_serialize)) for p in payloads
        ]
        self._payload_idx = 0
        ### END CLAUDE ###
        self.max_steps = max_steps

        self._ws: WebSocketApp | None = None
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
        self._send_count = 0
        self._recv_count = 0
        self._last_recv_t = time.time()
        self._last_send_t = time.time()

    # --- WS event handlers ---
    def _on_open(self, ws):
        print(f"[dummy] Connected to {self.server_url}")
        self._connected.set()

    def _on_message(self, ws, message):
        now = time.time()
        dt = now - self._last_recv_t
        self._last_recv_t = now
        try:
            data = json.loads(message)
            version = data.get("version", -1)
            action_data = data.get("action")
            action = convert_numpy_in_dict(action_data, numpy_deserialize)
            self._recv_count += 1
            if isinstance(action, np.ndarray):
                print(
                    f"[dummy] recv #{self._recv_count} v={version} "
                    f"shape={action.shape} dt={dt*1000:.1f}ms "
                    f"first3={action.flatten()[:3].tolist()}"
                )
            else:
                print(f"[dummy] recv non-array payload: {action!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[dummy] error parsing message: {e}")

    def _on_error(self, ws, error):
        print(f"[dummy] ws error: {error}")

    def _on_close(self, ws, code, reason):
        print(f"[dummy] connection closed: {code} {reason}")
        self._stop.set()

    # --- send loop runs in its own thread ---
    def _send_loop(self):
        if not self._connected.wait(timeout=10.0):
            print("[dummy] timed out waiting for connection")
            self._stop.set()
            return

        next_tick = time.perf_counter()
        while not self._stop.is_set():
            try:
                with self._send_lock:
                    if not (self._ws and self._ws.sock and self._ws.sock.connected):
                        print("[dummy] socket dropped; stopping send loop")
                        break
                    ### CLAUDE ### Cycle through the episode's observations; wrap at the end.
                    # self._ws.send(self.payload_json)
                    obs_idx = self._payload_idx
                    self._ws.send(self.payload_jsons[obs_idx])
                    self._payload_idx = (self._payload_idx + 1) % len(self.payload_jsons)
                    ### END CLAUDE ###
                self._send_count += 1
                now = time.time()
                print(
                    ### CLAUDE ### Include the obs index so we can see the loop cycling.
                    # f"[dummy] send #{self._send_count} dt={(now-self._last_send_t)*1000:.1f}ms"
                    f"[dummy] send #{self._send_count} obs_idx={obs_idx}/{len(self.payload_jsons)} "
                    f"dt={(now-self._last_send_t)*1000:.1f}ms"
                    ### END CLAUDE ###
                )
                self._last_send_t = now
                if self.max_steps is not None and self._send_count >= self.max_steps:
                    print(f"[dummy] reached max_steps={self.max_steps}, stopping")
                    self._stop.set()
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[dummy] send error: {e}")
                break

            next_tick += OBS_SEND_INTERVAL
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()

        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass

    def run(self):
        self._ws = WebSocketApp(
            self.server_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        sender = threading.Thread(target=self._send_loop, daemon=True)
        sender.start()
        try:
            self._ws.run_forever()
        except KeyboardInterrupt:
            print("[dummy] KeyboardInterrupt, exiting")
        finally:
            self._stop.set()
            sender.join(timeout=2.0)
            print(f"[dummy] done. sent={self._send_count} recv={self._recv_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(DEFAULT_RUN_DIR),
        help="Training run dir (must match the one serving the policy).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after sending this many obs; default runs until interrupted.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Override the instruction (default: use the dataset's first task).",
    )
    args = parser.parse_args()

    ### CLAUDE ### Load the entire first episode and cycle through it instead of repeating one obs.
    # payload = build_first_observation(args.run_dir)
    # if args.instruction is not None:
    #     payload["instruction"] = args.instruction
    #     print(f"[dummy] overriding instruction -> '{args.instruction}'")
    payloads = build_first_episode_observations(args.run_dir)
    if args.instruction is not None:
        for p in payloads:
            p["instruction"] = args.instruction
        print(f"[dummy] overriding instruction -> '{args.instruction}'")
    ### END CLAUDE ###

    server_url = f"ws://{args.host}:{args.port}/ws"
    ### CLAUDE ### Pass the list of episode payloads to the client.
    # client = DummyRTCClient(server_url, payload, max_steps=args.max_steps)
    client = DummyRTCClient(server_url, payloads, max_steps=args.max_steps)
    ### END CLAUDE ###
    client.run()


if __name__ == "__main__":
    main()
