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
import json
import os
import sys
import time
import pickle
import shutil
from datetime import datetime, timedelta, timezone
from typing import cast

import bittensor as bt
import uvloop

import tplr
from neurons import BaseNode

INFO_LEVEL = 2
TIME_LEVEL = 3
SUCCESS_LEVEL = 5
WARNING_LEVEL = 6

class LightweightDownloader(BaseNode):
    def log_with_level(self, message: str, level: int = 0):
        tplr.logger.info(f"\033[{97 - level}m{message}\033[0m")

    # Command line config items.
    @staticmethod
    def miner_config():
        parser = argparse.ArgumentParser(description="Simple Miner script")
        parser.add_argument(
            "--netuid", type=int, default=3, help="Bittensor network UID."
        )
        parser.add_argument(
            "--project", type=str, default="templar", help="Wandb project."
        )
        parser.add_argument(
            "--device", type=str, default="cpu", help="Device to use (CPU only)"
        )
        parser.add_argument(
            "--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0))
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        parser.add_argument(
            "--test",
            action="store_true",
            help="Test mode - use all peers without filtering",
        )
        parser.add_argument(
            "--delay-seconds",
            type=int,
            default=30,
            help="Delay in seconds before submitting gradient",
        )
        parser.add_argument(
            "--download-uid",
            type=int,
            default=None,
            help="UID to download gradient from (if not specified, will use random peer)",
        )
        parser.add_argument(
            "--gradient-storage-dir",
            type=str,
            default="./gradient_storage",
            help="Directory to store downloaded gradients",
        )
        parser.add_argument(
            "--max-gradients",
            type=int,
            default=6,
            help="Maximum number of different gradients to store",
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

    def __init__(self):
        tplr.logger.debug("Starting lightweight downloader initialization...")

        # Init config
        self.config = LightweightDownloader.miner_config()

        config_data = self.load_config_from_file("myconfig.json")

        self.config.wallet.name = config_data["wallet.name"]
        self.config.wallet.hotkey = config_data["wallet.hotkey"]

        # Set device to CPU
        self.device = "cpu"
        self.config.device = "cpu"

        # Load hparams (simplified)
        self.hparams = tplr.load_hparams(use_local_run_hparams=False)

        # Init bittensor objects
        self.wallet = bt.wallet(config=self.config)

        tplr.logger.info("[Init] Bittensor wallet loaded")
        super().__init__()

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

        self.log_with_level(f"{self.uid}: {self.wallet.hotkey.ss58_address}", SUCCESS_LEVEL)

        self.bucket = self.comms.get_own_bucket("gradients", "read")
        self.comms.try_commit(self.wallet, self.bucket)

        # Init state params
        self.current_block = self.comms.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        tplr.logger.info(
            f"[Init] chain at block {self.current_block}, window {self.current_window}"
        )

        self.start_window = self.current_window
        self.global_step = 0
        self.comms.current_window = self.current_window

        # Initialize peer related attributes
        self.next_peers: list[int] | None = None
        self.next_reserve_peers: list[int] | None = None
        self.peers_update_window = -1

        # Setup gradient storage
        self.gradient_storage_dir = self.config.gradient_storage_dir
        self.max_gradients = self.config.max_gradients
        self.current_gradient_index = 0

        
        self.ckpt = tplr.DCPCheckpointer(
            comms=self.comms,
            uid=self.uid,
            version=tplr.__version__,
        )

        # Create storage directory
        os.makedirs(self.gradient_storage_dir, exist_ok=True)
        
        # Clean old gradients on startup
        # self.cleanup_old_gradients()

        self.number = {248:1, 213:2, 55:3, 212:4, 227:5, 69:6}

        self.log_with_level("[Init] ✔ Lightweight downloader ready – entering run()", SUCCESS_LEVEL)

    async def download_gradient_from_peer(self, window: int, uid: int) -> dict | None:
        """Download gradient from a specific peer for the given window"""
        try:
            self.log_with_level(f"Downloading gradient from UID {uid} for window {window}", INFO_LEVEL)
            
            result = await self.comms.get(
                uid=str(uid),
                window=window,
                key="gradient",
                local=False,
                stale_retention=100,
            )
            
            if result.success and result.data is not None:
                self.log_with_level(f"Successfully downloaded gradient from UID {uid}", SUCCESS_LEVEL)
                return result.data
            else:
                self.log_with_level(f"Failed to download gradient from UID {uid}: {result.message}", WARNING_LEVEL)
                return None
                
        except Exception as e:
            self.log_with_level(f"Error downloading gradient from UID {uid}: {e}", WARNING_LEVEL)
            return None

    async def get_available_peers(self, window: int) -> list[int]:
        """Get list of available peers for gradient download"""
        try:
            # Get all UIDs from metagraph except self
            all_uids = list(range(1, len(self.comms.metagraph.S)))
            available_peers = [uid for uid in all_uids if uid != self.uid and uid != self.uid2]
            
            if self.config.test:
                self.log_with_level("Test mode: Using all peers from metagraph", INFO_LEVEL)
                return available_peers
            
            # Filter peers based on stake or other criteria if needed
            return available_peers[:10]  # Limit to first 10 peers
            
        except Exception as e:
            self.log_with_level(f"Error getting available peers: {e}", WARNING_LEVEL)
            return []

    def cleanup_old_gradients(self):
        """Clean up old gradient files to free disk space"""
        try:
            gradient_files = [f for f in os.listdir(self.gradient_storage_dir) if f.startswith("gradient_")]
            if len(gradient_files) > self.max_gradients:
                # Sort by modification time and remove oldest
                gradient_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.gradient_storage_dir, x)))
                files_to_remove = gradient_files[:-self.max_gradients]
                for file in files_to_remove:
                    file_path = os.path.join(self.gradient_storage_dir, file)
                    os.remove(file_path)
                    self.log_with_level(f"Removed old gradient file: {file}", INFO_LEVEL)
        except Exception as e:
            self.log_with_level(f"Error cleaning up old gradients: {e}", WARNING_LEVEL)

    def save_gradient_to_disk(self, gradient: dict, gradient_index: int) -> str:
        """Save gradient to disk and return file path"""
        file_path = os.path.join(self.gradient_storage_dir, f"gradient_{gradient_index}.pkl")
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(gradient, f)
            self.log_with_level(f"Saved gradient to {file_path}", SUCCESS_LEVEL)
            return file_path
        except Exception as e:
            self.log_with_level(f"Error saving gradient to disk: {e}", WARNING_LEVEL)
            return None

    def load_gradient_from_disk(self, gradient_index: int) -> dict | None:
        """Load gradient from disk"""
        file_path = os.path.join(self.gradient_storage_dir, f"gradient_{gradient_index}.pkl")
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    gradient = pickle.load(f)
                self.log_with_level(f"Loaded gradient from {file_path}", SUCCESS_LEVEL)
                return gradient
            else:
                self.log_with_level(f"Gradient file not found: {file_path}", WARNING_LEVEL)
                return None
        except Exception as e:
            self.log_with_level(f"Error loading gradient from disk: {e}", WARNING_LEVEL)
            return None

    def get_available_gradient_files(self, uid: int) -> list[int]:
        """Get list of available gradient file indices"""
        try:
            gradient_files = [f for f in os.listdir(self.gradient_storage_dir) if f.startswith("gradient_")]
            indices = []
            for file in gradient_files:
                try:
                    index = int(file.replace("gradient_", "").replace(".pkl", ""))
                    if self.number[uid] == index: # Only get the gradient file for the given uid
                        indices.append(index)
                except ValueError:
                    continue
            return sorted(indices)
        except Exception as e:
            self.log_with_level(f"Error getting available gradient files: {e}", WARNING_LEVEL)
            return []

    async def download_and_save_gradient(self, window: int):
        """Download gradient from a peer and save it to disk"""
        # Get available peers
        peers = await self.get_available_peers(window)
        if not peers:
            self.log_with_level("No peers available for gradient download", WARNING_LEVEL)
            return

        # Try to download from specified UID or random peer
        target_uid = 33
        # if target_uid is None or target_uid not in peers:
        #     import random
        #     target_uid = random.choice(peers)

        # Download gradient
        print(window - 1, target_uid)
        count = 6
        i = 0
        while count > 0:
            gradient = await self.download_gradient_from_peer(window - i, target_uid)
            if i > 30:
                break
            if gradient is not None:
                # print(gradient.keys())
                # Save to disk
                file_path = self.save_gradient_to_disk(gradient, self.current_gradient_index)
                count -= 1
                if file_path:
                    self.current_gradient_index = (self.current_gradient_index + 1) % self.max_gradients
                    self.log_with_level(f"Gradient saved successfully, next index: {self.current_gradient_index}", SUCCESS_LEVEL)
                else:
                    self.log_with_level("Failed to save gradient to disk", WARNING_LEVEL)
            else:
                self.log_with_level("Failed to download gradient, will retry next window", WARNING_LEVEL)
            i += 1

    async def submit_gradient_with_delay(self, window: int, delay_seconds: int, uid: int, first: bool = False):
        """Load gradient from disk and submit with a delay"""
        # Get available gradient files
        config_data = self.load_config_from_file("myconfig.json")
        submit_uids = config_data["submit_uid"]

        # Select a gradient to submit (rotate through available ones)
        for i, uid in enumerate(submit_uids):
            gradient_index = self.number[uid] - 1
            gradient = self.load_gradient_from_disk(gradient_index)
            print(f"Gradient loaded for UID {uid} at index {gradient_index}")

            self.log_with_level(f"Waiting {delay_seconds} seconds before submitting gradient {gradient_index}...", INFO_LEVEL)
            if i == 0:
                await asyncio.sleep(delay_seconds)       
            try:
                # Submit first gradient
                await self.comms.put(
                    state_dict=gradient,
                    uid=str(uid),
                    window=window,
                    key="gradient",
                    global_step=self.global_step,
                    local=False,
                    stale_retention=100,
                )
                tplr.logger.info(f"{tplr.T()} Submitting gradient {gradient_index} for UID {uid}")
                self.log_with_level(f"Submitted gradient {gradient_index} for UID {uid}", SUCCESS_LEVEL)

            except Exception as e:
                self.log_with_level(f"Error submitting gradient: {e}", WARNING_LEVEL)

    # Main loop.
    async def run(self):
        self.loop = asyncio.get_running_loop()

        # Use config peers if provided
        if self.config.peers:
            self.comms.peers = self.config.peers

        self.comms.commitments = await self.comms.get_commitments()
        tplr.logger.info("Loaded commitments")

        # Fetch peers
        start_window = await self.comms.get_start_window()
        tplr.logger.info(f"Using start_window: {start_window}")

        val = -1 if start_window is None else start_window
        start_window = None if val == -1 else int(val)
        assert start_window is not None
        self.start_window = start_window

        self.global_step = 0
        window_offset = self.current_window - (self.start_window or self.current_window)

        self.log_with_level(f"Starting with global_step=0, window offset={window_offset}", INFO_LEVEL)
        latest_window = None
        is_restart = False

        while not self.stop_event.is_set():
            # Initialize window
            window_start = tplr.T()
            step_window = self.current_window
            self.global_step = self.current_window - self.start_window
            
            self.log_with_level(
                f"\n{'-' * 40} Window: {step_window} (Global Step: {self.global_step}) {'-' * 40}", INFO_LEVEL
            )

            # Log timing
            window_total_time = tplr.T() - window_start
            self.log_with_level(
                f"Window {step_window} completed in {window_total_time:.2f}s", TIME_LEVEL
            )

            # Download aggregator if it exists
            retries = 10

            # Check if aggregator file already exists; if so, skip download
            aggregator_dir = os.path.join(self.ckpt.repo_root, "aggregator")
            os.makedirs(aggregator_dir, exist_ok=True)
            version = getattr(self.ckpt, "version", tplr.__version__)
            aggregator_path = os.path.join(
                aggregator_dir, f"{version}-{step_window - 1}.aggregator"
            )
            if os.path.exists(aggregator_path):
                self.log_with_level(f"Aggregator already exists at {aggregator_path}, skipping download.", SUCCESS_LEVEL)
                retries = -1
            else:
                retries = 10

            while retries > 0:
                tplr.logger.info(f"Downloading aggregator with retries {retries}...")
                fetch = await self.comms.get(
                    uid=str(1),
                    window=step_window - 1,
                    key="aggregator",
                    local=False,
                    stale_retention=100,
                )
                # tplr.logger.info(f"Fetch: {fetch}")
                if fetch.success and fetch.data is not None and "state_dict" in fetch.data:
                    self.log_with_level("Downloaded aggregator", SUCCESS_LEVEL)
                # Save fetch to the local aggregator/{version}-{window}.aggregator
                if fetch.data is not None and "state_dict" in fetch.data:
                    aggregator_dir = os.path.join(self.ckpt.repo_root, "aggregator")
                    print(f"aggregator_dir: {aggregator_dir}")
                    os.makedirs(aggregator_dir, exist_ok=True)
                    version = getattr(self.ckpt, "version", tplr.__version__)
                    print(f"version: {version}")
                    aggregator_path = os.path.join(
                        aggregator_dir, f"{version}-{step_window - 1}.aggregator"
                    )
                    print(f"aggregator_path: {aggregator_path}")
                    try:
                        with open(aggregator_path, "wb") as f:
                            pickle.dump(fetch.data, f)
                        self.log_with_level(f"Aggregator saved to {aggregator_path}", SUCCESS_LEVEL)
                        break
                    except Exception as e:
                        self.log_with_level(f"Failed to save aggregator: {e}", WARNING_LEVEL)
                        retries -= 1
                else:
                    self.log_with_level("Failed to download aggregator", WARNING_LEVEL)
                    retries -= 1
                await asyncio.sleep(60)
            if retries > 0:
                self.log_with_level("Downloaded aggregator", SUCCESS_LEVEL)
            elif retries == 0:
                self.log_with_level("Failed to download aggregator", WARNING_LEVEL)
            else:
                self.log_with_level("Skipped aggregator download", INFO_LEVEL)

            # Download checkpoint if it exists
            tplr.logger.info("Download checkpoint...")
            retries = 10
            # Find latest checkpoint's window which is saved locally
            checkpoints_root = os.path.join(self.ckpt.repo_root, f"checkpoints/{tplr.__version__}")
            _latest_window = None
            if os.path.exists(checkpoints_root):
                windows = []
                for name in os.listdir(checkpoints_root):
                    if name.isdigit():
                        windows.append(int(name))
                if windows:
                    _latest_window = max(windows)
            
            tplr.logger.info(f"Latest window: {_latest_window}")
            while retries > 0:
                is_new_checkpoint = False
                
                tplr.logger.info(f"Trying to download checkpoint with retries {retries}...")
                latest_window = await self.ckpt._discover_latest(prefer_highest_staked=True)
                tplr.logger.info(f"Latest window: {latest_window}")
                tplr.logger.info(f"Check latest_window is changed: latest_window: {latest_window}, _latest_window: {_latest_window}")
                if _latest_window == latest_window:
                    self.log_with_level(f"Latest window is the same, skipping checkpoint download. latest_window: {latest_window}, _latest_window: {_latest_window}", INFO_LEVEL)
                    await asyncio.sleep(50)
                    retries -= 1
                    continue
                self.log_with_level(f"Latest window: {latest_window}", INFO_LEVEL)

                if latest_window is not None:
                    tplr.logger.info(f"Downloading checkpoint for window: {latest_window}")
                    await self.ckpt.download_distributed(
                        window=latest_window,
                        prefer_highest_staked=True
                    )

                    # Check that all required checkpoint files exist
                    checkpoint_dir = os.path.join(
                        self.ckpt.repo_root, f"checkpoints/{tplr.__version__}/{latest_window}"
                    )
                    required_files = [
                        "__0_0.distcp",
                        "__1_0.distcp",
                        "__2_0.distcp",
                        "__3_0.distcp",
                        ".metadata",
                        "extra_metadata.json",
                    ]
                    missing_files = []
                    for fname in required_files:
                        fpath = os.path.join(checkpoint_dir, fname)
                        if not os.path.exists(fpath):
                            missing_files.append(fname)
                    if missing_files:
                        self.log_with_level(
                            f"Missing checkpoint files for window {latest_window}: {', '.join(missing_files)}",
                            WARNING_LEVEL
                        )
                    else:
                        self.log_with_level(
                            f"All checkpoint files exist for window {latest_window}",
                            SUCCESS_LEVEL
                        )

                    self.log_with_level("Downloaded checkpoint", SUCCESS_LEVEL)
                    
                    # After successful download, remove all previously downloaded checkpoints
                    # to free up storage and avoid redundancy.
                    for i in range(latest_window - 1000, latest_window - 1):
                        checkpoint_path = os.path.join(self.ckpt.repo_root, f"checkpoints/{tplr.__version__}/{i}")
                        if os.path.exists(checkpoint_path):
                            try:
                                shutil.rmtree(checkpoint_path)
                                is_new_checkpoint = True
                            except Exception as e:
                                self.log_with_level(f"Failed to remove old checkpoint: {e}", WARNING_LEVEL)
                            self.log_with_level(f"Removed old checkpoint: {i}", INFO_LEVEL)
                    for i in range(latest_window - 1000, latest_window):
                        aggregator_path = os.path.join(self.ckpt.repo_root, f"aggregator/{tplr.__version__}-{i}.aggregator")
                        if os.path.exists(aggregator_path):
                            try:
                                if os.path.isdir(aggregator_path):
                                    shutil.rmtree(aggregator_path)
                                else:
                                    os.remove(aggregator_path)
                            except Exception as e:
                                self.log_with_level(f"Failed to remove old aggregator: {e}", WARNING_LEVEL)
                            self.log_with_level(f"Removed old aggregator: {i}", INFO_LEVEL)
                    print(f"is_new_checkpoint: {is_new_checkpoint}")

                    break
                else:
                    tplr.logger.info("No checkpoint found")
                    retries -= 1

            # Wait for next window
            tplr.logger.info(f"Waiting for next window... {step_window + 1}")
            await self.wait_until_window(step_window + 1)
            if is_restart:
                tplr.logger.info("Waiting for 200 seconds...")
                await asyncio.sleep(200)
                self.log_with_level("Restarting miner", INFO_LEVEL)
                os.system("pm2 stop 0 && sleep 20 && pm2 start 0")

                # Load configuration from auto.json
                try:
                    with open("auto.json", "r") as f:
                        auto_config = json.load(f)
                    self.log_with_level(f"Loaded auto.json: {auto_config}", INFO_LEVEL)

                    # change run_id order. pop last and insert that to first
                    # This block rotates the "run_id" list in auto.json by moving the last element to the front.
                    if "run_id" in auto_config and isinstance(auto_config["run_id"], list) and auto_config["run_id"]:
                        last = auto_config["run_id"].pop()
                        auto_config["run_id"].insert(0, last)
                        self.log_with_level(f"Reordered run_id (workspace): {auto_config['run_id']}", INFO_LEVEL)
                    is_restart = False
                except Exception as e:
                    self.log_with_level(f"Failed to load auto.json: {e}", WARNING_LEVEL)

                self.log_with_level("Restarted miner", SUCCESS_LEVEL)
            tplr.logger.info(f"{tplr.T() - window_start} Completed waiting for next window")
            await asyncio.sleep(100)

def load_config_from_file(file_path: str):
    config_data = {}
    with open(file_path, "r") as f:
        config = json.load(f)
        config_data.update(config)
    return config_data

# Start lightweight downloader.
if __name__ == "__main__":
    uvloop.install()
    try:
        asyncio.run(LightweightDownloader().main())
    except KeyboardInterrupt:
        pass
