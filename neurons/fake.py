# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Standard library
import argparse
import asyncio
import os
import pickle
import concurrent.futures
import gc
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast

import bittensor as bt
import numpy as np
import torch
import uvloop
from torch.amp.grad_scaler import GradScaler
from torch.distributed.tensor import DTensor as DT

import tplr
from neurons import BaseNode, Trainer
from neurons.base_node import CPU_COUNT
from tplr import model_factory

# GPU optimizations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

INFO_LEVEL = 2
TIME_LEVEL = 3
SUCCESS_LEVEL = 5
WARNING_LEVEL = 6

class Miner(BaseNode, Trainer):
    def log_gpu_memory(self, stage: str):
        """Log current GPU memory allocation and reservation"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            tplr.logger.info(
                f"[GPU Memory - {stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )

    def check_memory_threshold(self, threshold_gb: float = 0.5):
        """Check if available memory is below threshold and cleanup if needed"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            max_memory = (
                torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            )
            available = max_memory - allocated

            if available < threshold_gb:
                tplr.logger.warning(f"Low GPU memory: {available:.2f} GB available")
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
                self.log_gpu_memory("After emergency cleanup")

    # Command line config items.
    @staticmethod
    def miner_config():
        parser = argparse.ArgumentParser(description="Miner script")
        parser.add_argument(
            "--netuid", type=int, default=3, help="Bittensor network UID."
        )
        parser.add_argument(
            "--project", type=str, default="templar", help="Wandb project."
        )
        parser.add_argument(
            "--actual-batch-size",
            type=int,
            default=None,
            help="Override the batch size defined in hparams.",
        )
        parser.add_argument(
            "--device", type=str, default="cpu", help="Device to use for training"
        )
        parser.add_argument(
            "--amp-dtype",
            choices=["bf16", "fp16"],
            default="bf16",
            help="Mixed-precision data type. Use «fp16» to enable GradScaler.",
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        parser.add_argument(
            "--store-gathers",
            action="store_true",
            help="Store gathered gradients in R2",
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Test mode - use all peers without filtering",
        )
        parser.add_argument(
            "--local",
            action="store_true",
            help="Local run - use toy model, small enough for a laptop.",
        )
        parser.add_argument(
            "--profile-iters",
            type=int,
            default=0,
            help="Active iterations per Torch‑Profiler trace (0 = disable)",
        )
        parser.add_argument(
            "--profile-dir",
            type=str,
            default="./log/profiler",
            help="Directory to save profiler traces",
        )
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()

        return config

    def load_config_from_file(self, file_path: str) -> dict:
        try:
            with open(file_path, "r") as f:
                config_data = json.load(f)
            return config_data
        except FileNotFoundError:
            tplr.logger.error(f"CRITICAL: Config file not found at {file_path}")
            raise
        except Exception as e:
            tplr.logger.error(f"Error loading {file_path}: {e}")
            raise

    def log_with_level(self, message: str, level: int = 0):
        tplr.logger.info(f"\033[{97 - level}m{message}\033[0m")

    def __init__(self):
        tplr.logger.debug("Starting initialization...")

        # Init config and load hparams
        self.config = Miner.miner_config()
        # ---------------------------------------------------------------------
        # Distributed initialisation
        # ---------------------------------------------------------------------

        # Mixed precision setup
        self.amp_dtype = (
            torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        )
        self.scaler = GradScaler(
            enabled=(self.amp_dtype is torch.float16 and self.device.type == "cuda")
        )
        tplr.logger.info(
            f"[Init] Using {self.config.amp_dtype}. GradScaler enabled: {self.scaler.is_enabled()}"
        )

        self.hparams = tplr.load_hparams(use_local_run_hparams=self.config.local)

        if self.config.actual_batch_size is not None:
            tplr.logger.info(
                f"Overriding hparams batch size: {self.hparams.batch_size} -> {self.config.actual_batch_size}"
            )
            self.hparams.batch_size = self.config.actual_batch_size

        # Init bittensor objects
        self.wallet = bt.wallet(config=self.config)
        tplr.logger.info("[Init] Bittensor wallet loaded")
        super().__init__()

        self.bootstrap_version = getattr(self.hparams, "checkpoint_init_version", None)
        tplr.logger.info(
            f"[Miner] code_version={tplr.__version__} "
            f"checkpoint_init_flag={self.bootstrap_version or '<none>'}"
        )

        # Init comms
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=self.config,
            hparams=self.hparams,
            uid=None,  # UID will be set after comms is initialized
        )

        if self.wallet.hotkey.ss58_address not in self.comms.metagraph.hotkeys:
            tplr.logger.error(
                f"\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.comms.metagraph.netuid}[/bold]"
            )
            sys.exit()
        self.uid = self.comms.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.comms.uid = self.uid

        self.ckpt = tplr.DCPCheckpointer(
            self.comms, uid=self.uid, version=tplr.__version__, repo_root="."
        )

        self.bucket = self.comms.get_own_bucket("gradients", "read")
        self.comms.try_commit(self.wallet, self.bucket)

        # Init state params
        self.current_block = self.comms.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        tplr.logger.info(
            f"[Init] chain at block {self.current_block}, window {self.current_window}"
        )

        self.start_window = self.current_window  # Record the start window
        self.global_step = 0  # Initialize global_step to zero
        self.comms.current_window = self.current_window
        self.step_counter = 0

        # Track additional metrics
        self.total_tokens_processed = 0

        # Initialize peer related attributes
        self.next_peers: list[int] | None = None
        self.next_reserve_peers: list[int] | None = None
        self.peers_update_window = -1

        self.log_with_level("[Init] ✔ fully done – entering run()", SUCCESS_LEVEL)

    # Main training loop.
    async def run(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        self.comms.commitments = await self.comms.get_commitments()
        tplr.logger.info("Loaded commitments")

        peer_start = tplr.T()
        # Fetch peers and get start_window from highest stake validator

        start_window = await self.comms.get_start_window()
        tplr.logger.info(f"Using start_window: {start_window}")
        self.start_window = start_window

        # global_step tracks actual outer steps performed (starts at 0)
        self.global_step = 0

        self.comms.start_commitment_fetcher()

        tplr.logger.info(
            f"Starting with global_step={self.global_step} (actual outer steps)"
        )

        def _gradient_local_path(id):
            return f"/gradient_storage/gradient_{id}.pkl"

        # Ensure gradient storage directory exists before any read/write
        storage_dir = os.path.dirname(_gradient_local_path(0))
        if storage_dir and not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)

        myconfig = self.load_config_from_file("myconfig.json")
        submit_uids = myconfig["submit_uid"]
        download_uid, download_window = 9, self.current_window
        for i in range(6):
            local_gradient_path = _gradient_local_path(i)
            if os.path.exists(local_gradient_path):
                tplr.logger.info(f"Gradient {i} exists in local file")
            else:
                tplr.logger.info(f"Downloading gradient for uid={download_uid}, window={download_window}")
                download_window -= 1
                while True:
                    gradient = await self.comms.get(
                        uid=str(download_uid),
                        window=download_window,
                        key="gradient",
                        local=False,
                    )
                    if gradient.success and isinstance(gradient.data, dict):
                        with open(local_gradient_path, "wb") as f:
                            f.write(gradient.data if isinstance(gradient.data, bytes) else pickle.dumps(gradient.data))
                        tplr.logger.info(f"Saved downloaded gradient to {local_gradient_path}")
                        break
                    else:
                        tplr.logger.warning(f"No gradient found for uid={download_uid}, window={download_window}")
                    download_window -= 1

        while not self.stop_event.is_set():
            await asyncio.sleep(0)
            # 1. Initialize window and update peers
            # Start the gather in the background:
            step_window = self.current_window
            # global_step will be incremented only when we do an actual outer step
            tplr.logger.info(
                f"\n{'-' * 40} Window: {step_window} (Outer Steps Taken: {self.global_step}) {'-' * 40}"
            )

            # 2. Load data
            config_data = self.load_config_from_file("myconfig.json")
            submit_uids = config_data["submit_uid"]

            num = {248: 0, 213: 1, 55: 2, 212: 3, 227: 4, 69: 5}

            for uid in submit_uids:
                local_gradient_path = _gradient_local_path(num[uid])
                if os.path.exists(local_gradient_path):
                    with open(local_gradient_path, "rb") as f:
                        gradient = pickle.load(f)
                    tplr.logger.info(f"Loaded gradient from local file: {local_gradient_path}")
                else:
                    download_window = self.window - 1
                    tplr.logger.info(f"Local gradient is missing, downloading for uid = {num[uid]}")
                    while True:
                        gradient = await self.comms.get(
                            uid=str(num[uid]),
                            window=download_window,
                            key="gradient",
                            local=False,
                        )
                        if gradient.success and isinstance(gradient.data, dict):
                            with open(local_gradient_path, "wb") as f:
                                pickle.dump(gradient.data, f)
                            tplr.logger.info(f"Saved downloaded gradient to {local_gradient_path}")
                            break
                        else:
                            tplr.logger.warning(f"No gradient found for uid={num[uid]}, window={download_window}")
                        download_window -= 1

            if self.current_window == step_window:
                tplr.logger.info(
                    "Training complete; waiting for window to be exhausted..."
                )
                await self.wait_until_window(step_window + 1)

            for i, uid in enumerate(submit_uids):
                local_gradient_path = _gradient_local_path(num[uid])
                with open(local_gradient_path, "rb") as f:
                    gradient = pickle.load(f)
                tplr.logger.info(f"Loaded gradient for uid={uid} from {local_gradient_path}")
                if i == 0:
                    await asyncio.sleep(60)
                    tplr.logger.info(f"Waiting for 60 seconds before uploading gradient for uid={uid}")
                    await self.comms.put(
                        state_dict=gradient,
                        uid=str(uid),
                        window=step_window,
                        key="gradient",
                        local=False,
                        stale_retention=100,
                    )
                    tplr.logger.info(f"Uploaded gradient for uid={uid}, window={step_window}")
                else:
                    tplr.logger.info(f"Skipping upload: no gradient to upload for {uid}")

            tplr.logger.info("Wait for next window...")
            await self.wait_until_window(step_window + 1)

    async def cleanup_window(self):
        """Aggressive memory cleanup between windows"""
        # Clear gradients more thoroughly
        self.model.zero_grad(set_to_none=True)
        self.inner_optimizer.zero_grad(set_to_none=True)

        # Clear error feedback for non-owned params to save memory
        for name in list(self.error_feedback.keys()):
            if name not in self.owned_params and self.error_feedback[name] is not None:
                self.error_feedback[name] = None

        # Clear any cached autocast states
        torch.clear_autocast_cache()

        # Empty CUDA cache multiple times for thorough cleanup
        for _ in range(3):
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

        # Force garbage collection
        gc.collect()

        # Check memory threshold after cleanup
        self.check_memory_threshold(threshold_gb=1.0)

        # Log memory status
        tplr.logger.info(
            f"After cleanup - GPU allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB"
        )
        tplr.logger.info(
            f"After cleanup - GPU reserved: {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB"
        )


# Start miner.
if __name__ == "__main__":
    uvloop.install()
    try:
        asyncio.run(Miner().main())
    except KeyboardInterrupt:
        pass
