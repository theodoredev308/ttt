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

# GPU optimizations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class FakeSubmitter(BaseNode, Trainer):
    def __init__(self, uid: int, window: int, config=None):
        self.uid = uid
        self.window = window
        self.config = config or self._default_config()
        self.wallet = bt.wallet(config=self.config)
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=self.config,
            hparams=None,
            uid=self.uid,
        )

    def _default_config(self):
        # Minimal config for wallet/comms
        import argparse
        parser = argparse.ArgumentParser()
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        return bt.config(parser)

    async def run(self):
        tplr.logger.info(f"FakeSubmitter started for uid={self.uid}, window={self.window}")
        # Download gradient for this uid and window
        import os
        import pickle

        def _gradient_local_path(id):
            return f"/gradient_storage/gradient_{id}.pkl"

        # Ensure gradient storage directory exists before any read/write
        storage_dir = os.path.dirname(_gradient_local_path(0))
        if storage_dir and not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)

        # List of UIDs to process (6 UIDs: self.uid, self.uid+1, ..., self.uid+5)
        import json

        # Load UIDs from myconfig.json file
        myconfig_path = "myconfig.json"
        with open(myconfig_path, "r") as f:
            myconfig = json.load(f)

        # Use the submit_uid list from config, or default to [self.uid, self.uid+1, ..., self.uid+5]
        if myconfig.get("submit_uid"):
            uids = myconfig["submit_uid"]
        else:
            uids = [248, 213, 55, 212, 227, 69]

        # Download or load gradients for all UIDs for the current window
        gradient = None
        download_uid = 9
        download_window = self.window
        for i in range(6):
            local_gradient_path = _gradient_local_path(i)
            if os.path.exists(local_gradient_path):
                tplr.logger.info(f"Loading gradient from local file: {i}")
                # with open(local_gradient_path, "rb") as f:
                    # gradients[i] = pickle.load(f)
            else:
                tplr.logger.info(f"Downloading gradient for uid={download_uid}, window={self.window}")
                download_window -= 1
                while True:
                    gradient = await self.comms.get(
                        uid=str(download_uid),
                        window=download_window,
                        key="gradient",
                        local=False,
                    )
                    if (
                        not gradient.success
                        or not isinstance(gradient.data, dict)
                    ):
                        tplr.logger.warning(f"No gradient found for uid={i}, window={self.window}")
                        continue
                    else:
                        with open(local_gradient_path, "wb") as f:
                            pickle.dump(gradient.data, f)
                        tplr.logger.info(f"Saved downloaded gradient to {local_gradient_path}")
                        break
                    download_window -= 1

        while True:
            # Always try to load the gradients from local files for all UIDs
            num = {248: 0, 213: 1, 55: 2, 212: 3, 227: 4, 69: 5}
            with open("myconfig.json", "r") as f:
                myconfig = json.load(f)
            submit_uids = myconfig["submit_uid"]
            for id in submit_uids:
                local_gradient_path = _gradient_local_path(num[id])
                if os.path.exists(local_gradient_path):
                    with open(local_gradient_path, "rb") as f:
                        gradient = pickle.load(f)
                    tplr.logger.info(f"Loaded gradient from local file: {local_gradient_path}")
                else:
                    download_window = self.window - 1
                    tplr.logger.info(f"Local gradient missing, downloading for uid={download_uid}, window={download_window}")
                    while True:
                        gradient = await self.comms.get(
                            uid=str(num[id]),
                            window=download_window,
                            key="gradient",
                            local=False,
                        )
                        if (
                            not gradient.success
                            or not isinstance(gradient.data, dict)
                        ):
                            tplr.logger.warning(f"No gradient found for uid={download_uid}, window={download_window}")
                        else:
                            with open(local_gradient_path, "wb") as f:
                                pickle.dump(gradient, f)
                            tplr.logger.info(f"Downloaded and saved gradient to {local_gradient_path}")
                            break
                        download_window -= 1
            
            # Wait 60 seconds before uploading
            if self.current_window == self.window:
                tplr.logger.info("Waiting for window to be exhausted...")
                await self.wait_until_window(self.window + 1)

            # Upload the gradients (if any) for all UIDs for this window
            for i, uid in enumerate(submit_uids):
                local_gradient_path = _gradient_local_path(i)
                with open(local_gradient_path, "rb") as f:
                    gradient = pickle.load(f)
                tplr.logger.info(f"Loaded gradient for uid={uid} from {local_gradient_path}")
                if gradient is not None:
                    if i == 0:
                        await asyncio.sleep(60)
                    tplr.logger.info(f"Uploading gradient for uid={uid}, window={self.window}")
                    await self.comms.put(
                        state_dict=gradient,
                        uid=str(uid),
                        window=self.window,
                        key="gradient",
                        local=False,
                        stale_retention=100,
                    )
                    tplr.logger.info(f"Uploaded gradient for uid={uid}, window={self.window}")
                else:
                    tplr.logger.warning(f"Skipping upload: no gradient to upload for uid={uid}, window={self.window}")

            # Wait until next window
            tplr.logger.info("Waiting for next window...")
            await self._wait_until_next_window()
            self.window += 1
            # Update local_gradient_path for the new window (not strictly needed, handled in loop)

    async def _wait_until_next_window(self):
        # Wait until the chain advances to the next window
        subtensor = self.comms.subtensor
        blocks_per_window = getattr(self.comms.hparams, "blocks_per_window", 100) if self.comms.hparams else 100
        current_block = subtensor.block
        current_window = int(current_block / blocks_per_window)
        target_window = self.window + 1
        target_block = target_window * blocks_per_window
        while subtensor.block < target_block:
            await asyncio.sleep(5)
        tplr.logger.info(f"Advanced to window {target_window}")

if __name__ == "__main__":
    uvloop.install()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=int, required=True, help="UID to download/upload gradient for")
    parser.add_argument("--window", type=int, required=True, help="Window to start from")
    args = parser.parse_args()
    try:
        asyncio.run(FakeSubmitter(uid=args.uid, window=args.window).run())
    except KeyboardInterrupt:
        pass
