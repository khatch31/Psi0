import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

### CLAUDE ### model_validator needed to override ServerConfig.set_policy for the multi-ckpt case
# from pydantic import BaseModel
from pydantic import BaseModel, model_validator
### END CLAUDE ###
from collections import deque
import threading
import time
import copy

import os
import sys
import json
import tyro
# import draccus
import uvicorn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List
import os.path as osp
import pickle
from datetime import datetime
### CLAUDE ### Multi-checkpoint hot-swap support
from dataclasses import dataclass
### END CLAUDE ###

from rich.console import Console
console = Console()
def color_print(*args, markup=False, style="red"):
    console.print(*args, style=style, markup=markup)

from psi.models.psi0 import Psi0Model 

# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure imports work regardless of current working directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from psi.config.config import LaunchConfig, ServerConfig
from psi.deploy.helpers import *

# from pipelines import ActionPipeline
# from misc import move_to_device
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything

from psi.utils.overwatch import initialize_overwatch 
overwatch = initialize_overwatch(__name__)



PREDICT_HORIZON = 30          # == H
MIN_EXEC_HORIZON = 15         # == s_min # TODO: should match D_INIT, ideally s_min >= d_real
DELAY_BUFFER_SIZE = 6        # == delay_buffer_size
D_INIT = 6                   # == d_init # TODO: placeholder, needs calculation
### CLAUDE ### Control period is now supplied via the --ctrl_period_sec CLI arg instead of being
### hardcoded, so the same server can drive a 30Hz or a 10Hz checkpoint. The constant below is
### kept only as the fallback default when the arg is not passed.
# CTRL_PERIOD_SEC = 1. / 30       # 30Hz
DEFAULT_CTRL_PERIOD_SEC = 1. / 30       # 30Hz
### END CLAUDE ###

PSI_HOME = os.environ["PSI_HOME"]

color_print("PSI_HOME:", PSI_HOME, style="cyan")

import sys
from IPython.core.debugger import Pdb
class ForkedIPdb(Pdb):
    """An ipdb subclass that can be used from a forked multiprocessing child."""

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            super().interaction(*args, **kwargs)
        finally:
            sys.stdin = _stdin

### CLAUDE ### Clean process exit that does not swallow the last thing we printed.
def hard_exit(code: int = 0):
    """os._exit skips atexit handlers AND stdio flushing, so buffered output is lost. Flush first,
    otherwise the final message printed before exiting never reaches the terminal."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


### Device-move helper for checkpoint hot-swapping.
### Psi0Model.from_pretrained sets `model.device` as a PLAIN ATTRIBUTE (src/psi/models/psi0.py:1603)
### and never calls .to() itself. nn.Module.to() moves parameters/buffers but knows nothing about
### that attribute, so a bare model.to("cpu") leaves model.device pointing at cuda. The model reads
### self.device everywhere it allocates a tensor (psi0.py:1681, 1715, 1756, 1765, 1795, 1829, ...),
### so a stale value means inputs land on a different device than the weights. Always move via this
### helper -- never call model.to(...) directly on a policy.
def move_policy_to_device(model, device):
    dev = torch.device(device)
    model.to(dev)
    model.device = dev
    return model


@dataclass
class CheckpointSlot:
    """One loadable policy, plus every per-checkpoint config derived from its launch_config.

    A checkpoint is more than weights: maxmin (action/state normalization stats), model_transform
    (image resize/crop) and the launch config drive _parse_obs_payload and _postprocess_action.
    Swapping weights without swapping these would denormalize the new policy's actions with the old
    policy's statistics -- plausible-looking but wrong motion on the robot. Keeping them bundled per
    slot makes it impossible to swap one without the others.
    """
    label: str
    run_dir: Path
    ckpt_step: int
    model: Any
    launch_cfg: Any
    maxmin: Any
    model_transform: Any
    Da: int
    Tp: int
    Ta: int

    @property
    def on_gpu(self) -> bool:
        return self.model.device.type == "cuda"
### END CLAUDE ###


class RealTimeChunkController:
    def __init__(self,
                 policy: Psi0Model,
                 prediction_horizon: int = PREDICT_HORIZON,
                 min_exec_horizon: int = MIN_EXEC_HORIZON,
                 delay_buf_size: int = DELAY_BUFFER_SIZE,
                 d_init: int = D_INIT,
                 o_first: np.ndarray | None = None,
                 timestamp: str = None, # type: ignore
                 ### CLAUDE ### Label the active checkpoint so saved inference data is attributable
                 ### to the policy that produced it across a swap.
                 ckpt_label: str = "ckpt",
                 ### END CLAUDE ###
                 ):

        self.policy : Psi0Model = policy
        ### CLAUDE ### NOTE: this caches the device a second time (the first is model.device). A
        ### swap must build a FRESH controller rather than reassigning self.policy in place, or this
        ### cache goes stale. See Server._activate_slot.
        self.device = self.policy.device
        ### END CLAUDE ###
        self.H     = prediction_horizon
        ### CLAUDE ### Used to build save paths; see _save_dir()
        self.ckpt_label = ckpt_label
        ### END CLAUDE ###

        ### CLAUDE ### Serialize all diffusion sampling. The inference-loop thread and the
        ### reset() executor thread both call into self.policy, which shares a single stateful
        ### noise_scheduler (_step_index). Concurrent use races: one thread's set_timesteps()
        ### resets _step_index to None while another thread's step() does `_step_index += 1`,
        ### causing "unsupported operand type(s) for +=: 'NoneType' and 'int'". This lock makes
        ### prediction mutually exclusive across threads. Defined before the warmup calls below.
        self._predict_lock = threading.Lock()
        ### END CLAUDE ###

        ### CLAUDE ### Stop flag so the inference thread can be torn down on a checkpoint swap.
        ### Must be defined before the warmup calls below and before _infer_th starts.
        self._stop = threading.Event()
        ### END CLAUDE ###

        self.s_min = min_exec_horizon

        self.t: int = 0
        self.inference_counter: int = 0
        self.start_timestamp = time.time()

        # from datetime import datetime
        # self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.timestamp = timestamp
        color_print(f"[RealTimeChunkController] self.timestamp: {self.timestamp}", style="green")
        assert o_first != None, "please provide o_first"

        A_first = self._predict_action(o_first) # (H, D)

        # warmup the model
        for i in range (2):
            _ = self._predict_action_rtc(copy.deepcopy(o_first), np.concatenate([copy.deepcopy(A_first[self.s_min:, :]), np.zeros((self.s_min, A_first.shape[1]), dtype=A_first.dtype)], axis=0), d_init, self.t)
        print("Model warmed up")

        self.A_cur = A_first # (H, D)
        self.o_cur: Dict[str, Any] | None = None 

        self.Q = deque([d_init], maxlen=delay_buf_size)

        ### CLAUDE ### Store d_init and delay_buf_size for use in reset()
        self.d_init = d_init
        self.delay_buf_size = delay_buf_size
        ### END CLAUDE ###

        self.M = threading.Lock()
        self.C = threading.Condition(self.M)

        self._infer_th = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_th.start()

    def replace_prev_actions_to_obs(self, o, previous_rpy, previous_height):
        o['obs'] = np.concatenate([o['obs'][:, :, :28], previous_rpy[np.newaxis, np.newaxis, :], previous_height[np.newaxis, np.newaxis, :], o['obs'][:, :, 32:]], axis=-1) # (1, 1, 28) -> (1, 1, 32)
        return o

        
    def step(self, obs_next: Dict[str, Any]): # consume a_(t-1) and provide o_t
        with self.C:
            self.t += 1
            self.o_cur = obs_next
            self.C.notify()
            if self.t-1 >= len(self.A_cur):
                single_action = self.A_cur[-1]
                print("failed")
            else:
                single_action = self.A_cur[self.t - 1]
            return single_action[np.newaxis, :] # (1, D)

    def _inference_loop(self):
        color_print("Started inference loop...", style="yellow")
        ### CLAUDE ### Loop on the stop flag so the thread can be torn down for a checkpoint swap.
        ### The inner wait gains a timeout so a stopped thread still wakes even if nothing notifies.
        # while True:
        while not self._stop.is_set():
        ### END CLAUDE ###
            with self.C:
                try:
                    ### CLAUDE ### Bail out of the wait when stopped
                    # while self.t < self.s_min:
                    #     self.C.wait() # wait until notified and get the lock
                    while self.t < self.s_min and not self._stop.is_set():
                        self.C.wait(timeout=0.2) # wait until notified and get the lock
                    if self._stop.is_set():
                        break
                    ### END CLAUDE ###
                    s   = self.t

                    # FIXME: 
                    # 1. maybe bug at "s-2"
                    # 2. inputs should be : normalize_states(denormalize_action(A_cur[s-2])) 
                    #    but in our current data, the stats for rpy and height are nearly the same, 
                    #    so "normalize_states(denormalize_action())" equals to doing nothing.

                    assert (s-2) >= 0
                    self.o_cur = self.replace_prev_actions_to_obs(self.o_cur, copy.deepcopy(self.A_cur[s-2, 28:31]), copy.deepcopy(self.A_cur[s-2, 31:32]))
                    #

                    o   = copy.deepcopy(self.o_cur)
                    
                    d   = max(self.Q)
                    # d   = min(max(self.Q), 7) ### DUMMY CLIENT ###
                    # A_prev = copy.deepcopy(torch.cat([self.A_cur[s:, :], torch.zeros((s, self.A_cur.shape[1]), device=self.A_cur.device, dtype=self.A_cur.dtype)], dim=0)) # (H, D)
                    A_prev = np.concatenate([copy.deepcopy(self.A_cur[s:, :]), np.zeros((s, self.A_cur.shape[1]), dtype=self.A_cur.dtype)], axis=0) # (H, D)

                    inference_start = time.perf_counter()
                    self.C.release()
                    # color_print(f"\ns: {s}", style="yellow")
                    # color_print(f"o: {o}", style="yellow")
                    # state = o['obs']
                    # color_print(f"state.shape: {state.shape}", style="yellow")
                    # color_print(f"d: {d}", style="yellow")  
                    A_new = self._predict_action_rtc(o, A_prev, d, s)
                    # color_print(f"A_new: {A_new}", style="yellow")
                    self.C.acquire()

                    self.A_cur = A_new
                    self.t = self.t - s
                    self.Q.append(self.t)          
                    # self.C.notify_all()
                    print(f"[inference]  latency={time.perf_counter()-inference_start:.4f}s  s={s}  d={d}  self.t={self.t}")
                except Exception as e:
                    ### CLAUDE ### A stopped controller is an intentional teardown (checkpoint
                    ### swap), not a crash. Tearing the model off the GPU underneath an in-flight
                    ### prediction can raise here; killing the process for that would make every
                    ### swap fatal.
                    if self._stop.is_set():
                        color_print(f"[RealTimeChunkController] inference loop exiting during teardown ({e})", style="yellow")
                        return
                    ### END CLAUDE ###
                    print(f"\n[ERROR] Inference loop crashed!")
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print("\n[FATAL] Stopping program...")
                    os._exit(1)  # 强制退出整个程序

        ### CLAUDE ### Normal exit path for a swap
        color_print("[RealTimeChunkController] inference loop stopped", style="yellow")
        ### END CLAUDE ###
    
    def _predict_action_rtc(self, o, A_prev, d, s):
        
        ### CLAUDE ### Hold the predict lock around the policy call so the inference-loop thread
        ### and the reset() thread never use the shared noise_scheduler concurrently.
        with self._predict_lock:
            timestamp = time.time() - self.start_timestamp

            A_new = self.policy.predict_action_with_training_rtc_flow(
                        observations=o['imgs'],
                        # states=torch.from_numpy(o['obs']).to(self.device),
                        states=torch.from_numpy(np.zeros_like(o['obs'])).to(self.device), ### ZERO OUT ###
                        traj2ds=None,
                        instructions=o['text_instructions'],
                        num_inference_steps = 8,
                        # num_inference_steps = 4, ### DUMMY CLIENT ###
                        # prev_actions=torch.from_numpy(A_prev[np.newaxis, :, :]).to(self.device), # (H, D) -> (1, H, D)
                        prev_actions=torch.from_numpy(A_prev[np.newaxis, :, :-2]).to(self.device), # (H, D) -> (1, H, D)
                        inference_delay=d,
                        max_delay=8
                    )[0].float().detach().cpu().numpy() # (1, H, D) -> (H, D)
            ### END CLAUDE ###

            # A_new[..., -8:] = 0.0 ### ZERO OUT ###

            ### CLAUDE ### Namespace saved inference by checkpoint label so data before and after a
            ### swap is attributable to the policy that produced it.
            # save_dir = Path(f"{PSI_HOME}/saved_inference/{self.timestamp}/policy_time_inference")
            save_dir = Path(f"{PSI_HOME}/saved_inference/{self.timestamp}/{self.ckpt_label}/policy_time_inference")
            ### END CLAUDE ###
            obs_dir = save_dir / "observations"
            act_dir = save_dir / "actions"
            img_dir = save_dir / "images"
            obs_dir.mkdir(parents=True, exist_ok=True)
            act_dir.mkdir(parents=True, exist_ok=True)
            img_dir.mkdir(parents=True, exist_ok=True)

            obs_file = obs_dir / f"obs_{s}_{self.inference_counter}.pkl"
            actions_file = act_dir / f"actions_{s}_{self.inference_counter}.npy"

            

            
            timestamp_arr = np.array([timestamp for _ in range(A_new.shape[0])]).astype(A_new.dtype)
            timestamp_arr = np.expand_dims(timestamp_arr, axis=-1)
            inference_counter_arr = np.array([self.inference_counter for _ in range(A_new.shape[0])]).astype(A_new.dtype)
            inference_counter_arr = np.expand_dims(inference_counter_arr, axis=-1)

            A_new = np.concatenate([A_new, inference_counter_arr, timestamp_arr], axis=-1)

            o["timestamp"] = timestamp
            o["inference_counter"] = self.inference_counter


            with open(obs_file, 'wb') as f:
                pickle.dump(o, f)

            # with open(actions_file, 'wb') as f:
            #     pickle.dump(A_new, f)
            np.save(actions_file, A_new)


            # [[<PIL.Image.Image image mode=RGB size=320x240 at 0x7B752C177CA0>]]
            # o["imgs"][0][0].size
            images = o["imgs"]

            for batch_idx, batch in enumerate(images):
                for img_idx, img in enumerate(batch):
                    img_file = img_dir / f"img_{s}_{self.inference_counter}_batch{batch_idx}_img{img_idx}.png"
                    img.save(img_file)

            self.inference_counter += 1

            return A_new
    
    def _predict_action(self, o):
        

        ### CLAUDE ### Hold the predict lock around the policy call so the inference-loop thread
        ### and the reset() thread never use the shared noise_scheduler concurrently.
        with self._predict_lock:
            timestamp = time.time() - self.start_timestamp

            normalized_actions = self.policy.predict_action(
                        observations=o['imgs'],
                        # states=torch.from_numpy(o['obs']).to(self.device),
                        states=torch.from_numpy(np.zeros_like(o['obs'])).to(self.device), ### ZERO OUT ###
                        traj2ds=None,
                        instructions=o['text_instructions'],
                        num_inference_steps = 8,
                    )[0].float().detach().cpu().numpy() # (1, H, D) -> (H, D)
            ### END CLAUDE ###

            # normalized_actions[..., -8:] = 0.0 ### ZERO OUT ###
            
            ### CLAUDE ### Namespace saved inference by checkpoint label so data before and after a
            ### swap is attributable to the policy that produced it.
            # save_dir = Path(f"{PSI_HOME}/saved_inference/{self.timestamp}/policy_time_inference")
            save_dir = Path(f"{PSI_HOME}/saved_inference/{self.timestamp}/{self.ckpt_label}/policy_time_inference")
            ### END CLAUDE ###
            obs_dir = save_dir / "observations"
            act_dir = save_dir / "actions"
            img_dir = save_dir / "images"
            obs_dir.mkdir(parents=True, exist_ok=True)
            act_dir.mkdir(parents=True, exist_ok=True)
            img_dir.mkdir(parents=True, exist_ok=True)

            obs_file = obs_dir / f"obs_initial_{self.inference_counter}.pkl"
            actions_file = act_dir / f"actions_initial_{self.inference_counter}.npy"

            timestamp_arr = np.array([timestamp for _ in range(normalized_actions.shape[0])]).astype(normalized_actions.dtype)
            timestamp_arr = np.expand_dims(timestamp_arr, axis=-1)
            inference_counter_arr = np.array([self.inference_counter for _ in range(normalized_actions.shape[0])]).astype(normalized_actions.dtype)
            inference_counter_arr = np.expand_dims(inference_counter_arr, axis=-1)

            normalized_actions = np.concatenate([normalized_actions, inference_counter_arr, timestamp_arr], axis=-1)

            o["timestamp"] = timestamp
            o["inference_counter"] = self.inference_counter

            with open(obs_file, 'wb') as f:
                pickle.dump(o, f)

            # with open(actions_file, 'wb') as f:
            #     pickle.dump(normalized_actions, f)
            np.save(actions_file, normalized_actions)

            # [[<PIL.Image.Image image mode=RGB size=320x240 at 0x7B752C177CA0>]]
            # o["imgs"][0][0].size
            images = o["imgs"]

            ### CLAUDE ### Save images to PNG files for debugging/analysis
            for batch_idx, batch in enumerate(images):
                for img_idx, img in enumerate(batch):
                    img_file = img_dir / f"img_initial_{self.inference_counter}_batch{batch_idx}_img{img_idx}.png"
                    img.save(img_file)
            ### END CLAUDE ### 


            self.inference_counter += 1

            return normalized_actions

    ### CLAUDE ### Reset controller: clear history and get fresh action prediction using _predict_action (no history)
    def reset(self, o_new: Dict[str, Any]):
        """
        Reset controller state. Runs _predict_action (no RTC history) on the provided observation
        to produce a fresh A_cur, then resets t, o_cur, and Q so the inference loop starts clean.
        Safe to call from any thread.
        """
        color_print("[RealTimeChunkController] Reset: running fresh prediction...", style="cyan")
        # Run inference outside the lock — _predict_action is slow
        A_fresh = self._predict_action(o_new)

        with self.C:
            self.A_cur = A_fresh
            self.t = 0
            self.o_cur = None
            self.Q = deque([self.d_init], maxlen=self.delay_buf_size)
            self.C.notify_all()

        color_print("[RealTimeChunkController] Reset complete", style="cyan")
    ### END CLAUDE ###

    ### CLAUDE ### Tear down the inference thread so a checkpoint swap can retire this controller.
    def stop(self, join_timeout: float = 10.0):
        """Signal the inference thread to exit and wait for it.

        Safe to call from any thread. After this returns, no further calls will be made into
        self.policy, which is what makes it safe to move the model off the GPU.
        """
        self._stop.set()
        with self.C:
            self.C.notify_all()

        self._infer_th.join(timeout=join_timeout)
        if self._infer_th.is_alive():
            color_print(
                f"[RealTimeChunkController] WARNING: inference thread still alive after "
                f"{join_timeout}s; it is likely inside a diffusion call. Waiting on the predict "
                f"lock before it is safe to move the model.",
                style="red",
            )

        # Acquiring the predict lock guarantees any in-flight sampling has finished, even if the
        # join timed out. Without this a swap could move weights mid-prediction.
        with self._predict_lock:
            pass
        return not self._infer_th.is_alive()
    ### END CLAUDE ###


class Server:

    def __init__(
        self,
        policy:str,
        ### CLAUDE ### Multi-checkpoint: take a list of (run_dir, ckpt_step) pairs instead of one.
        ### The first is activated on the GPU; the rest stay parked in host RAM.
        # run_dir: Path,
        # ckpt_step: int | str  = "latest",
        ckpt_specs: List[Any] = None, # list of (Path, int)
        ### END CLAUDE ###
        device: str = "cuda:0",
        enable_rtc: bool = False,
        action_exec_horizon: int | None = None,
        ### CLAUDE ### Control period now passed in rather than read from a module constant
        ctrl_period_sec: float = DEFAULT_CTRL_PERIOD_SEC
        ### END CLAUDE ###
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

        self.device = torch.device(device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {policy}")

        ### CLAUDE ### Load every checkpoint to CPU, then activate the first onto the GPU.
        assert ckpt_specs, "at least one checkpoint is required (--ckpt RUN_DIR:STEP)"

        self.action_exec_horizon_override = action_exec_horizon
        self.swap_lock = threading.RLock()

        self.slots: List[CheckpointSlot] = []
        for idx, (run_dir, ckpt_step) in enumerate(ckpt_specs):
            overwatch.info(f"loading action model {idx + 1}/{len(ckpt_specs)} ...")
            self.slots.append(self._load_slot(Path(run_dir), ckpt_step))

        # Re-seed AFTER all loads. from_pretrained fully random-initializes a 2B-param VLM
        # (psi0.py:1534) before overwriting it via load_state_dict, so every load consumes global
        # RNG. Seeding only up front would make the first prediction depend on how many checkpoints
        # were passed and in what order -- the same --ckpt list reordered would produce different
        # actions. Seeding here makes runs comparable across different checkpoint lists.
        seed_everything(self.slots[0].launch_cfg.seed or 42)

        chunk_sizes = {s.Tp for s in self.slots}
        if len(chunk_sizes) > 1:
            color_print(
                f"[Server] WARNING: checkpoints disagree on action_chunk_size {sorted(chunk_sizes)}. "
                f"This session uses a single global ctrl_period_sec, so verify these were all "
                f"trained at the same control rate before swapping between them.",
                style="red",
            )

        self.active_idx: int = 0
        active = self.slots[0]
        move_policy_to_device(active.model, self.device)
        self._bind_active_slot(active)
        color_print(f"[Server] active checkpoint: {active.label} (on {active.model.device})", style="green")
        ### END CLAUDE ###

        self.count = 0

        ### CLAUDE ### Store the control period and log it loudly. The RTC constants
        ### (MIN_EXEC_HORIZON, D_INIT, DELAY_BUFFER_SIZE) are all in units of control ticks, so
        ### their wall-clock meaning scales with this value -- printing it makes the effective
        ### rate obvious at startup instead of something you have to infer from the tick logs.
        self.ctrl_period_sec = ctrl_period_sec
        color_print(
            f"[Server] ctrl_period_sec={self.ctrl_period_sec:.4f}s "
            f"({1.0 / self.ctrl_period_sec:.2f}Hz), "
            f"MIN_EXEC_HORIZON={MIN_EXEC_HORIZON} ticks "
            f"(~{MIN_EXEC_HORIZON * self.ctrl_period_sec:.2f}s between replans)",
            style="green",
        )
        ### END CLAUDE ###


        # control - shared state with locks
        self.latest_obs = None
        self.latest_action = None
        self.action_version = 0  # Used by client to check if there's a new action
        
        self.obs_lock = threading.Lock()
        self.action_lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.inference_counter = 0

        ### CLAUDE ### Gate for the control loop. Set == running. Cleared during a checkpoint swap
        ### so no actions are emitted while the model is off the GPU / the controller is being
        ### rebuilt. The robot holds its last commanded pose for the duration.
        self._run_control = threading.Event()
        self._run_control.set()
        ### END CLAUDE ###

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        color_print(f"[Server] self.timestamp: {self.timestamp}", style="green")

        self.controller = None
        self._control_loop_started = False
        
        # WebSocket: asyncio event to notify when new action is ready
        self.app = FastAPI()
        self._setup_routes()
        
        self._action_ready_event: asyncio.Event = None  # Will be created in async context
        self._active_websocket: WebSocket = None
        self._loop = None  # asyncio event loop reference for thread-safe notification
        self.start_time = time.time()
        self.start_time_obs = time.time()

    ### CLAUDE ### Multi-checkpoint slot management
    def _load_slot(self, run_dir: Path, ckpt_step) -> CheckpointSlot:
        """Load one checkpoint plus everything derived from its launch_config, resting on CPU."""
        assert osp.exists(run_dir), f"run_dir {run_dir} does not exist!"
        assert osp.exists(run_dir / "checkpoints" / f"ckpt_{ckpt_step}"), f"ckpt {ckpt_step} does not exist in {run_dir}!"
        assert osp.exists(run_dir / "run_config.json"), f"run config does not exist in {run_dir}!"

        # first build dynamic config
        config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt") # type: ignore
        # then load it from previously saved json
        conf = (run_dir / "run_config.json").open("r").read()
        launch_config = config_.model_validate_json(conf)

        from psi.models.psi0 import Psi0Model
        # Load to CPU: from_pretrained already builds everything on CPU (psi0.py:1526, 1534) and
        # only records the device attribute, so CPU is its natural resting place. The active slot
        # is moved onto the GPU afterwards by the caller.
        model = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device="cpu")
        move_policy_to_device(model, "cpu")
        model.eval()

        from psi.config.transform import SimpleRepackTransform, Psi0ModelTransform, ActionStateTransform
        maxmin: ActionStateTransform = launch_config.data.transform.field # type:ignore
        model_transform: Psi0ModelTransform = launch_config.data.transform.model # type:ignore

        Tp = launch_config.model.action_chunk_size # type:ignore
        Ta = self.action_exec_horizon_override or launch_config.model.action_exec_horizon # type:ignore
        assert Ta <= Tp, f"action_exec_horizon {Ta} is too big for chunk size {Tp} ({run_dir})"

        # MIN_EXEC_HORIZON is in control ticks and bounds how far into a chunk the controller reads
        # before replanning. A chunk shorter than s_min means step() runs off the end of A_cur and
        # silently repeats the last action (the print("failed") path).
        assert MIN_EXEC_HORIZON < Tp, (
            f"MIN_EXEC_HORIZON={MIN_EXEC_HORIZON} must be < action_chunk_size={Tp} for {run_dir}, "
            f"otherwise the controller runs off the end of every chunk"
        )

        label = f"{run_dir.name.split('.')[0]}_step{ckpt_step}"
        slot = CheckpointSlot(
            label=label,
            run_dir=run_dir,
            ckpt_step=ckpt_step,
            model=model,
            launch_cfg=launch_config,
            maxmin=maxmin,
            model_transform=model_transform,
            Da=launch_config.model.action_dim, # type:ignore
            Tp=Tp,
            Ta=Ta,
        )
        color_print(f"[Server] loaded '{label}' -> cpu  (chunk={Tp}, exec_horizon={Ta})", style="cyan")
        return slot

    def _bind_active_slot(self, slot: CheckpointSlot):
        """Point the server's active-bundle attributes at `slot`.

        These must move together: _parse_obs_payload reads launch_cfg (image keys) and maxmin
        (state normalization); _postprocess_action reads maxmin (action denormalization). Binding
        new weights while leaving the old stats in place would denormalize the new policy's actions
        with the previous policy's statistics -- plausible-looking but wrong motion on the robot.
        """
        self.model = slot.model
        self.maxmin = slot.maxmin
        self.model_transform = slot.model_transform
        self.launch_cfg = slot.launch_cfg
        self.Da = slot.Da
        self.Tp = slot.Tp
        self.Ta = slot.Ta

    @property
    def active_slot(self) -> CheckpointSlot:
        return self.slots[self.active_idx]
    ### END CLAUDE ###

    def _init_controller(self, o_first):
        ### CLAUDE ### Pass the active label so saved inference is namespaced per checkpoint
        # controller = RealTimeChunkController(policy=self.model, o_first=o_first, timestamp=self.timestamp)
        controller = RealTimeChunkController(
            policy=self.model,
            o_first=o_first,
            timestamp=self.timestamp,
            ckpt_label=self.active_slot.label,
        )
        ### END CLAUDE ###
        return controller

    def _postprocess_action(self, action):
        # return self.launch_cfg.data.data_transforms.denormalize_action(action)
        return self.maxmin.denormalize(action) # denormalization is done in the pipeline



    def preprocess_image(self, image_dict: Dict[str, Any]) -> Dict[str, Any]:
        imgs = {}
        # # FIXME 
        # image_key_to_cam_idx = {'front_stereo_left': 0, 'front_stereo_right': 1, 'left_future_traj_2d': 4, 'right_future_traj_2d': 5, 'side': 3, 'side_future_traj_2d': 7, 'wrist': 2, 'wrist_future_traj_2d': 6}
        # for img_key in self.launch_config.data.transform.repack.image_keys:
        #     cam_idx = image_key_to_cam_idx[img_key] #self.launch_config.data.transform.repack.image_key_to_cam_idx[img_key]
        #     imgs[f"cam{cam_idx}"] = self._process_img(image_dict[f"{img_key}".replace("image_", "")])#[None, ...]

        for k in image_dict.keys():
            imgs[k] = self._process_img(image_dict[k])


        return imgs

    def _process_img(self, img):
        from torchvision.transforms import v2

        transforms = [self.model_transform.resize(), self.model_transform.center_crop()]
        t = v2.Compose(transforms)

        return [t(img)]

    def _parse_obs_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse observation payload and return processed obs dict"""
        request = RequestMessage.deserialize(payload)
        image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
                    request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name
        condition_dict = request.condition
        overwatch.info(f"Instruction: {instruction}")
            
        # parts = instruction.split(".")
        # if len(parts) > 1 and parts[-1].isdigit():
        #     instruction = parts[0].lower() + "."
        #     img_id = int(parts[-1])
        #     assert False
        # else:
        instruction = instruction.lower()
        # img_id = -1


        # TODO support image history
        # img dict: {"video": np.array(...).shape(480, 640, 3)}
        imgs = {}
        for cam_idx, img_key in enumerate(self.launch_cfg.data.transform.repack.image_keys):
            imgs[f"cam{cam_idx}"] = Image.fromarray(np.clip(image_dict[img_key], 0, 255).astype(np.uint8))
        
        hand_joints = state_dict["hand_joints"].copy() # shape (14,)
        arm_joints = state_dict["arm_joints"].copy() # shape (14,)
        tmp_torso_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        tmp_torso_height = np.array([0.75], dtype=np.float32)
        obs = np.concatenate([hand_joints, arm_joints, tmp_torso_rpy, tmp_torso_height], axis=-1) # (32,)

        # normalize states
        assert self.maxmin.normalize_state, "check"
        if self.maxmin.pad_state_dim != len(obs):
            obs = pad_to_len(obs, self.maxmin.pad_state_dim, dim=0)[0]
        obs = self.maxmin.normalize_state_func(obs) # shape (32,)
        obs = obs[np.newaxis, np.newaxis, :] # (32,) -> (1, 1, 32)


        image_input = self.preprocess_image(imgs)
        batch_images = [image_input['cam0']] # batch size == 1


        conditions = {}
        
        text_instructions = [instruction] # len == 1

        return {'imgs': batch_images, 'text_instructions': text_instructions, 'obs': obs, 'conditions': conditions}

    async def websocket_handler(self, websocket: WebSocket):
        """
        WebSocket handler for bidirectional communication:
        - Receive obs from client at high frequency
        - Send action to client immediately when new action is ready
        """
        await websocket.accept()
        self._active_websocket = websocket
        
        # Create asyncio event for action notification
        self._action_ready_event = asyncio.Event()
        
        print("[WebSocket] Client connected")
        async def receive_obs():
            """Continuously receive obs from client"""
            try:
                while True:
                    # Receive obs from client
                    data = await websocket.receive_text()
                    payload = json.loads(data)
                    interval = time.time() - self.start_time_obs
                    self.start_time_obs = time.time()
                    print(f"[WebSocket] receive_obs interval: {interval} seconds")
                    # Parse and update latest_obs
                    this_o = self._parse_obs_payload(payload)
                    with self.obs_lock:
                        self.latest_obs = this_o
                    
                    # If control loop hasn't started, start it
                    if not self._control_loop_started and self.latest_obs is not None:
                        self._start_control_loop()

                    # # 清空缓冲区，只保留最新的
                    # latest_data = None
                    # while True:
                    #     try:
                    #         # 非阻塞地读取所有可用消息
                    #         data = await asyncio.wait_for(
                    #             websocket.receive_text(), 
                    #             timeout=0.001  # 1ms超时
                    #         )
                    #         latest_data = data  # 保留最新的
                    #     except asyncio.TimeoutError:
                    #         break  # 没有更多消息了
                    
                    # if latest_data:
                    #     payload = json.loads(latest_data)
                    #     interval = time.time() - self.start_time_obs
                    #     self.start_time_obs = time.time()
                    #     print(f"[WebSocket] receive_obs interval: {interval} seconds")
                        
                    #     this_o = self._parse_obs_payload(payload)
                    #     with self.obs_lock:
                    #         self.latest_obs = this_o
                        
                    #     if not self._control_loop_started and self.latest_obs is not None:
                    #         self._start_control_loop()
                        
            except WebSocketDisconnect:
                print("[WebSocket] Client disconnected (receive)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[WebSocket] Receive error: {e}")
        
        async def send_action():
            """Send action to client when new action is ready"""
            try:
                while True:
                    # Wait for new action to be ready
                    await self._action_ready_event.wait()
                    self._action_ready_event.clear()

                    interval = time.time() - self.start_time
                    self.start_time = time.time()
                    print(f"[WebSocket] send_action interval: {interval} seconds")
                    
                    # Get the action
                    with self.action_lock:
                        action = self.latest_action
                        version = self.action_version
                        self.latest_action = None  # Reset after sending
                    
                    if action is not None:
                        # Send action to client
                        response = ResponseMessage(action, err=0.0)
                        resp_dict = response.serialize()
                        resp_dict["version"] = version
                        await websocket.send_text(json.dumps(resp_dict))
                        print(f"[WebSocket] Sent action, version={version}")
                    else:
                        assert False, "action is None"
                        
            except WebSocketDisconnect:
                print("[WebSocket] Client disconnected (send)")
            except Exception as e:
                print(f"[WebSocket] Send error: {e}")
        
        try:
            # Run both tasks concurrently
            await asyncio.gather(receive_obs(), send_action())
        except Exception as e:
            print(f"[WebSocket] Connection closed: {e}")
        finally:
            self._active_websocket = None
            print("[WebSocket] Handler finished")

    def _start_control_loop(self):
        """Start control loop thread"""
        if self._control_loop_started:
            return
        self._control_loop_started = True
        
        # Initialize controller with first obs
        with self.obs_lock:
            o_first = copy.deepcopy(self.latest_obs)
            
        self.controller = self._init_controller(o_first) # wait for model warm up
        
        # Start control loop thread
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        print("[control loop] started")

    ### CLAUDE ### Hot-swap which checkpoint owns the GPU.
    def swap_to(self, new_idx: int) -> bool:
        """Move the active policy to CPU, bring `new_idx` onto the GPU, rebuild the controller.

        The control loop is paused for the whole operation, so nothing is emitted while the model
        is in flight and the robot holds its last commanded pose. Fails closed: on any error the
        control loop stays paused rather than resuming with a half-swapped bundle.
        """
        with self.swap_lock:
            if new_idx == self.active_idx:
                color_print(f"[swap] '{self.active_slot.label}' is already active", style="yellow")
                return True

            old = self.active_slot
            new = self.slots[new_idx]
            color_print(f"[swap] {old.label} -> {new.label} ...", style="magenta")
            t0 = time.perf_counter()

            # 1. Pause the control loop. No further step() calls until we set this again.
            self._run_control.clear()

            try:
                # 2. Retire the old controller's inference thread. Once stop() returns, nothing is
                #    calling into old.model, which is what makes it safe to move.
                if self.controller is not None:
                    self.controller.stop()
                    self.controller = None

                # 3/4. Move the weights. move_policy_to_device also fixes up model.device
                #      (psi0.py:1603 sets it as a plain attribute that .to() does not maintain);
                #      without that the next prediction allocates its tensors on the wrong device.
                move_policy_to_device(old.model, "cpu")
                torch.cuda.empty_cache()
                move_policy_to_device(new.model, self.device)

                # 5. Re-point the active bundle: weights, normalization stats and image transforms
                #    all move together.
                self.active_idx = new_idx
                self._bind_active_slot(new)

                # 6. Rebuild the controller. This runs the warmup diffusion passes, which must
                #    happen now that the model is on the GPU -- the VLM uses flash_attention_2 and
                #    the sampling path hardcodes torch.autocast("cuda", ...) (psi0.py:1696 etc),
                #    neither of which works on a CPU-resident model.
                with self.obs_lock:
                    o_latest = copy.deepcopy(self.latest_obs)
                if o_latest is None:
                    color_print(
                        "[swap] no observation available to seed the new controller. Staying "
                        "paused; connect the client and swap again.",
                        style="red",
                    )
                    return False

                self.controller = self._init_controller(o_latest)

            except Exception as e:
                import traceback
                traceback.print_exc()
                color_print(
                    f"[swap] FAILED: {e}\n[swap] control loop left PAUSED -- the robot is holding "
                    f"position. Fix the cause and swap again, or quit.",
                    style="red",
                )
                return False

            # 7. Resume.
            self._run_control.set()
            color_print(
                f"[swap] complete in {time.perf_counter() - t0:.1f}s -- now serving '{new.label}'",
                style="green",
            )
            return True
    ### END CLAUDE ###

    def _control_loop(self):
        """
        Control loop: Execute controller.step strictly every CTRL_PERIOD_SEC
        And expect at next time, the obs_next sent from client is the one after executing the action
        """
        next_tick = time.perf_counter()
        prev_tick = time.perf_counter()
        
        while True:
            # loop_start = time.time()

            ### CLAUDE ### Pause point for checkpoint swaps. While cleared we emit nothing, so the
            ### robot holds its last commanded pose. On resume, rebase the tick clock so the paused
            ### interval is not reported as a long burst of missed ticks.
            if not self._run_control.is_set():
                color_print("[control loop] paused (checkpoint swap)", style="yellow")
                self._run_control.wait()
                next_tick = time.perf_counter()
                prev_tick = next_tick
                color_print("[control loop] resumed", style="yellow")

            # swap_to() briefly sets self.controller to None between retiring the old controller
            # and building the new one. Capture the reference once so we never dereference None.
            controller = self.controller
            if controller is None:
                time.sleep(0.01)
                continue
            ### END CLAUDE ###

            # 1. Get latest obs
            with self.obs_lock:
                obs_next = copy.deepcopy(self.latest_obs)

            # 2. Execute step
            ### CLAUDE ### Use the captured reference rather than re-reading self.controller
            # action = self.controller.step(obs_next) # (1, D)
            action = controller.step(obs_next) # (1, D)
            ### END CLAUDE ###
            action_info = action[:, 36:]
            # pred_action = self._postprocess_action(action) # (1, D)
            pred_action = self._postprocess_action(action[:, :36]) # (1, D)

            # color_print("\naction.shape:", action.shape, style="yellow")
            # color_print("action:", action, style="yellow")
            # color_print("pred_action.shape:", pred_action.shape, style="yellow")
            # color_print("pred_action:", pred_action, style="yellow")
            pred_action[..., -5] = 0.75 # need to hardcode 0.75 for torso height here? ### ZERO OUT ###
            # color_print("pred_action:", pred_action, style="yellow")

            with self.save_lock:
                if obs_next is not None:
                    ### CLAUDE ### Namespace by checkpoint label so data before and after a swap is
                    ### attributable to the policy that produced it.
                    # save_dir = Path(f"{PSI_HOME}/saved_inference/{self.timestamp}/deployment_time_inference")
                    save_dir = Path(f"{PSI_HOME}/saved_inference/{self.timestamp}/{self.active_slot.label}/deployment_time_inference")
                    ### END CLAUDE ###
                    obs_dir = save_dir / "observations"
                    act_dir = save_dir / "actions"
                    pred_act_dir = save_dir / "pred_actions"
                    act_info_dir = save_dir / "action_infos"
                    img_dir = save_dir / "images"
                    obs_dir.mkdir(parents=True, exist_ok=True)
                    act_dir.mkdir(parents=True, exist_ok=True)
                    pred_act_dir.mkdir(parents=True, exist_ok=True)
                    act_info_dir.mkdir(parents=True, exist_ok=True)
                    img_dir.mkdir(parents=True, exist_ok=True)

                    obs_file = obs_dir / f"obs_{self.inference_counter}.pkl"
                    actions_file = act_dir / f"actions_{self.inference_counter}.npy"
                    pred_actions_file = pred_act_dir / f"pred_actions_{self.inference_counter}.npy"
                    act_infos_file = act_info_dir / f"action_infos_{self.inference_counter}.npy"

                    with open(obs_file, 'wb') as f:
                        pickle.dump(obs_next, f)

                    np.save(actions_file, action)
                    np.save(pred_actions_file, pred_action)
                    np.save(act_infos_file, action_info)



                    # [[<PIL.Image.Image image mode=RGB size=320x240 at 0x7B752C177CA0>]]
                    # o["imgs"][0][0].size
                    images = obs_next["imgs"]

                    for batch_idx, batch in enumerate(images):
                        for img_idx, img in enumerate(batch):
                            img_file = img_dir / f"img_{self.inference_counter}_batch{batch_idx}_img{img_idx}.png"
                            img.save(img_file)

                    self.inference_counter += 1


            
            
            # 3. Update latest_action
            with self.action_lock:
                self.latest_action = pred_action
                self.action_version += 1
            
            # 4. Notify WebSocket that new action is ready
            if self._action_ready_event is not None:
                # Thread-safe way to set asyncio event from another thread
                try:
                    self._loop.call_soon_threadsafe(self._action_ready_event.set)
                except Exception as e:
                    print(f"[control loop] Failed to notify WebSocket: {e}")
            
            # elapsed = (time.time() - loop_start) * 1000
            # print(f"[control loop] step took {elapsed:.1f}ms, version={self.action_version}")
            
            # 5. Wait until next ctrl period
            ### CLAUDE ### Use the per-instance control period instead of the module constant
            # next_tick += CTRL_PERIOD_SEC
            next_tick += self.ctrl_period_sec
            ### END CLAUDE ###
            sleep_time = next_tick - time.perf_counter()
            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            print(f"[control loop] interval: {interval} seconds")
            if sleep_time > 0:
                time.sleep(sleep_time)
                # delay_ms(sleep_time*1000)
            else:
                print(f"[control loop] WARNING: missed tick by {-sleep_time*1000:.1f}ms")
                next_tick = time.perf_counter()
    

    def _setup_routes(self):
        """设置所有路由"""
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            self._loop = asyncio.get_event_loop()
            await self.websocket_handler(websocket)
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}

        ### CLAUDE ### POST /reset: clear history and latest obs, re-run fresh prediction from latest obs
        @self.app.post("/reset")
        async def reset_controller():
            with self.obs_lock:
                o_latest = copy.deepcopy(self.latest_obs)

            if self.controller is None:
                return {"status": "error", "message": "Controller not yet initialized — send at least one obs first"}
            if o_latest is None:
                return {"status": "error", "message": "No observation available to seed reset"}

            # Run blocking inference in a thread pool so we don't block the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.controller.reset, o_latest)

            # NOTE: do NOT clear latest_obs here — the control loop calls step(latest_obs)
            # every tick and would pass None into the controller, crashing the inference loop

            return {"status": "ok"}
        ### END CLAUDE ###

    ### CLAUDE ### Interactive checkpoint menu, entered on Ctrl+C.
    def _print_menu(self):
        print()
        color_print("=" * 72, style="magenta")
        color_print("  Checkpoint menu -- control loop PAUSED, robot holding last commanded pose", style="magenta")
        color_print("=" * 72, style="magenta")
        for i, slot in enumerate(self.slots):
            marker = "*" if i == self.active_idx else " "
            where = "gpu" if slot.on_gpu else "cpu"
            print(f"   {marker} [{i}] {slot.label:<44s} {where}   chunk={slot.Tp}")
        print()
        print("     <number>  swap to that checkpoint")
        print("     r         reset the active controller (fresh chunk, drop RTC history)")
        print("     c         continue with the current checkpoint")
        print("     q         quit")
        print()

    def _menu(self):
        """Blocking interactive menu, run on the main thread while the server keeps serving.

        Pauses the control loop on entry so the robot stops the moment Ctrl+C is pressed, rather
        than only once a swap is actually chosen.
        """
        was_running = self._run_control.is_set()
        self._run_control.clear()

        while True:
            self._print_menu()
            try:
                choice = input("   select> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Second Ctrl+C (or closed stdin) at the menu means quit.
                print()
                color_print("Exiting.", style="red")
                hard_exit(0)

            if choice in ("q", "quit", "exit"):
                color_print("Exiting.", style="red")
                hard_exit(0)

            if choice in ("c", "", "continue"):
                if was_running:
                    self._run_control.set()
                color_print(f"Continuing with '{self.active_slot.label}'.", style="green")
                return

            if choice in ("r", "reset"):
                if self.controller is None:
                    color_print("   no controller yet -- send at least one observation first", style="red")
                    continue
                with self.obs_lock:
                    o_latest = copy.deepcopy(self.latest_obs)
                if o_latest is None:
                    color_print("   no observation available to seed the reset", style="red")
                    continue
                self.controller.reset(o_latest)
                if was_running:
                    self._run_control.set()
                return

            if choice.isdigit() and 0 <= int(choice) < len(self.slots):
                if self.swap_to(int(choice)):
                    return
                # Swap failed -- swap_to left the control loop paused deliberately. Stay in the
                # menu so the operator can retry or quit rather than silently resuming.
                continue

            color_print(f"   unrecognized selection: {choice!r}", style="red")
    ### END CLAUDE ###

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        print(f"Server listens on {host}:{port}")
        print(f"WebSocket endpoint: ws://{host}:{port}/ws")

        ### CLAUDE ### Run uvicorn on a background thread with its signal handlers suppressed, so
        ### the main thread stays free to catch Ctrl+C and show the checkpoint menu. uvicorn.run()
        ### installs its own SIGINT handler that triggers graceful shutdown, which would make
        ### Ctrl+C kill the process instead of opening the menu. Suppressing them is also required
        ### because asyncio cannot install signal handlers from a non-main thread.
        # try:
        #     uvicorn.run(self.app, host=host, port=port)
        # except Exception as e:
        #     print(f"Server crashed, {e}")
        # finally:
        #     print("Server stopped.")
        #     exit(1)
        config = uvicorn.Config(self.app, host=host, port=port)
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        # If stdin is not a TTY (backgrounded, piped, nohup, CI) an input() prompt would block
        # forever with nobody to answer it, so fall back to plain SIGINT-exits behavior.
        interactive = sys.stdin.isatty()
        if interactive:
            color_print(
                f"[Server] {len(self.slots)} checkpoint(s) loaded. Press Ctrl+C for the checkpoint menu.",
                style="green",
            )
        else:
            color_print(
                "[Server] stdin is not a TTY -- the Ctrl+C checkpoint menu is disabled and SIGINT "
                "will exit. Run this in a terminal to use the menu.",
                style="yellow",
            )

        while server_thread.is_alive():
            try:
                time.sleep(0.5)
            except KeyboardInterrupt:
                if not interactive:
                    print()
                    color_print("Interrupted; exiting.", style="red")
                    hard_exit(0)
                self._menu()

        print("Server stopped.")
        hard_exit(0)
        ### END CLAUDE ###

### CLAUDE ### Local config so the shared ServerConfig -- used by eight other deploy servers -- is
### left untouched.
###
### --ckpt takes SPACE-SEPARATED values on a SINGLE flag, each pairing a run dir with its step:
###     --ckpt RUN_DIR_A:10000 RUN_DIR_B:38000
### Repeating the flag (--ckpt A --ckpt B) does NOT append -- tyro keeps only the last occurrence,
### which would silently serve just one checkpoint. This matches the existing list-argument
### convention in the repo (e.g. data.train_repo_ids in src/psi/config/data_lerobot.py:12).
class MultiCkptServerConfig(ServerConfig):
    ckpt: List[str] = []
    # Re-declared with defaults so the inherited required scalars become optional here. They are
    # still honored as a single-checkpoint fallback when --ckpt is not passed.
    run_dir: str = ""
    ckpt_step: int = 0

    @model_validator(mode="after")
    def set_policy(self):
        # Overrides ServerConfig.set_policy, which indexes run_dir.parts[1] and would raise when
        # run_dir is empty because checkpoints came in via --ckpt instead.
        if self.policy is None:
            src = self.ckpt[0].rpartition(":")[0] if self.ckpt else self.run_dir
            parts = Path(src).parts
            self.policy = parts[1] if len(parts) > 1 else "psi"
        return self


def parse_ckpt_specs(cfg: MultiCkptServerConfig):
    """Turn repeated --ckpt "RUN_DIR:STEP" entries into (Path, int) pairs."""
    specs = []
    for entry in cfg.ckpt:
        run_dir, sep, step = entry.rpartition(":")
        assert sep and step.isdigit(), (
            f"--ckpt entry {entry!r} must be RUN_DIR:STEP, e.g. /path/to/run_dir:10000"
        )
        specs.append((Path(run_dir), int(step)))

    if not specs:
        # Fallback: behave like the single-checkpoint server.
        assert cfg.run_dir, "pass at least one --ckpt RUN_DIR:STEP (or legacy --run-dir/--ckpt-step)"
        specs.append((Path(cfg.run_dir), cfg.ckpt_step))
    return specs
### END CLAUDE ###


### CLAUDE ### serve() now builds the server from a list of checkpoints
# def serve(cfg: ServerConfig) -> None:
def serve(cfg: MultiCkptServerConfig) -> None:
    overwatch.info("Server :: Initializing Policy")
    assert cfg.policy is not None, "which policy to serve?"
    assert cfg.rtc, "this server is for rtc"

    ckpt_specs = parse_ckpt_specs(cfg)
    color_print(f"[Server] {len(ckpt_specs)} checkpoint(s) requested:", style="cyan")
    for i, (run_dir, step) in enumerate(ckpt_specs):
        color_print(f"    [{i}] {run_dir}  step={step}", style="cyan")

    server = Server(
        cfg.policy,
        # Path(cfg.run_dir),
        # cfg.ckpt_step,
        ckpt_specs,
        cfg.device,
        cfg.rtc,
        cfg.action_exec_horizon,
        # Forward the control period from the CLI config
        cfg.ctrl_period_sec,
        )

    print("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # take environment variables from .env file
    # config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    config = tyro.cli(MultiCkptServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)
### END CLAUDE ###