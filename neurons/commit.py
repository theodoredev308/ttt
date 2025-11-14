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
import concurrent.futures
import gc
import hashlib
import json
import os
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
from tplr.distributed import dist_helper

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
            "--netuid", type=int, default=268, help="Bittensor network UID."
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
            "--device", type=str, default="cuda", help="Device to use for training"
        )
        parser.add_argument(
            "--amp-dtype",
            choices=["bf16", "fp16"],
            default="bf16",
            help="Mixed-precision data type. Use «fp16» to enable GradScaler.",
        )
        parser.add_argument(
            "--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0))
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
            self.old_myconfig = config_data
            return config_data
        except FileNotFoundError:
            tplr.logger.error(f"CRITICAL: Config file not found at {file_path}")
            config_data = self.old_myconfig
            tplr.logger.warning(f"Using old myconfig: {config_data}")
        except Exception as e:
            tplr.logger.error(f"Error loading {file_path}: {e}")
            config_data = self.old_myconfig
            tplr.logger.warning(f"Using old myconfig: {config_data}")
        return config_data

    def log_with_level(self, message: str, level: int = 0):
        tplr.logger.info(f"\033[{97 - level}m{message}\033[0m")

    def __init__(self):
        tplr.logger.debug("Starting initialization...")

        # Init config and load hparams
        self.config = Miner.miner_config()
        # ---------------------------------------------------------------------
        # Distributed initialisation
        # ---------------------------------------------------------------------

        # # Convenience flags - already set from dist_helper
        # self.config.local = cast(bool, self.config.local)
        self.hparams = tplr.load_hparams(use_local_run_hparams=self.config.local)

        # Init bittensor objects
        self.wallet = bt.wallet(config=self.config)
        tplr.logger.info("[Init] Bittensor wallet loaded")
        super().__init__()


        # # Init comms
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
        # if self.is_master:
        self.comms.try_commit(self.wallet, self.bucket)

    # Main training loop.
    async def run(self):
        return

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
