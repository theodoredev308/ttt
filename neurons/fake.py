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
import json
import sys
import time
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

class Fake(BaseNode):
    def log_with_level(self, message: str, level: int = 0):
        tplr.logger.info(f"\033[{97 - level}m{message}\033[0m")

    # Command line config items.
    @staticmethod
    def miner_config():
        parser = argparse.ArgumentParser(description="Fake Miner script")
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

    def log_with_level(self, message: str, level: int = 0):
        tplr.logger.info(f"\033[{97 - level}m{message}\033[0m")

    def __init__(self):
        tplr.logger.debug("Starting Fake miner initialization...")

        # Init config
        self.config = Fake.miner_config()

        config_data = self.load_config_from_file("myconfig.json")

        self.config.wallet.name = config_data["wallet.name"]
        self.config.wallet.hotkey = config_data["wallet.hotkey"]

        # Set device to CPU
        self.device = "cpu"
        self.config.device = "cpu"

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

        # Load wandb info
        self.wandb_info = tplr.WanDBInfo(
            entity=config_data["entity"],
            project=config_data["project"],
            run_id=config_data["run_id"]
        )

        # Setup gradient storage
        self.gradient_storage_dir = self.config.gradient_storage_dir
        self.max_gradients = self.config.max_gradients
        self.current_gradient_index = 0
        
        # Create storage directory
        os.makedirs(self.gradient_storage_dir, exist_ok=True)
        
        # Clean old gradients on startup
        self.cleanup_old_gradients()

        self.number = {248:1, 213:2, 55:3, 212:4, 227:5, 69:6}

        self.log_with_level("[Init] ✔ Simple miner ready – entering run()", SUCCESS_LEVEL)

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
            available_peers = [uid for uid in all_uids if uid != self.uid]
            
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
        target_uid = 9
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

    async def submit_gradient_with_delay(self, window: int, delay_seconds: int, first: bool = False):
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
        peer_start = tplr.T()
        await tplr.neurons.update_peers(
            instance=self, window=self.current_window, peer_start=peer_start
        )

        start_window = await self.comms.get_start_window()
        tplr.logger.info(f"Using start_window: {start_window}")

        assert start_window is not None
        self.start_window = start_window

        self.global_step = 0

        # Download initial gradient
        gradients_exist = all(
            os.path.exists(f"gradient_storage/gradient_{i}.pkl") for i in range(6)
        )
        print(f"Gradients exist: {gradients_exist}")
        if not gradients_exist:
            await self.download_and_save_gradient(self.current_window)

        # There is 6 files are ready so dont need to download again

        while not self.stop_event.is_set():
            await asyncio.sleep(0)
            
            # Initialize window
            window_start = tplr.T()
            step_window = self.current_window
            self.global_step = self.current_window - self.start_window
            
            self.log_with_level(
                f"\n{'-' * 40} Window: {step_window} (Global Step: {self.global_step}) {'-' * 40}", INFO_LEVEL
            )

            # Try to download new gradient (will save to disk)
            # Check if 6 gradients exist before downloading
            print(f"Step window: {step_window}")
            gradients_exist = all(
                os.path.exists(f"gradient_storage/gradient_{i}.pkl") for i in range(6)
            )
            print(f"Gradients exist: {gradients_exist}")
            if not gradients_exist:
                await self.download_and_save_gradient(step_window)

            # Log timing
            window_total_time = tplr.T() - window_start
            self.log_with_level(
                f"Window {step_window} completed in {window_total_time:.2f}s", TIME_LEVEL
            )

            # Wait for next window
            tplr.logger.info("Wait for next window...")
            await self.wait_until_window(step_window + 1)
            tplr.logger.info(f"{tplr.T() - window_start} Completed waiting for next window")

            # Submit gradient with delay
            my_config = self.load_config_from_file("myconfig.json")
            upload_start = my_config["upload_start"]
            if upload_start == 1:
                await self.submit_gradient_with_delay(step_window, 120, first=True)

            await self.wandb_sync()
            debug_result, debug_global_step = None, None
            sync_scores, sync_uids = self.wandb_info.get_sync_score(), self.wandb_info.get_sync_score_uids(20)
            self.log_with_level(f"Sync uids: {sync_uids}", INFO_LEVEL)

            temp_sync_uids = [int(uid) for uid in sync_uids if sync_scores[uid] == 1.0]
            if len(temp_sync_uids) == 0:
                tplr.logger.info("No uids are 1.0. just gathere from non-top")
                eval_uids = sync_uids[:10]
            else:
                eval_uids = temp_sync_uids[:10]

            self.log_with_level(f"Gathering from {eval_uids}", 1)
            count, loop_count = 0, 0
            debug_dict = {}
            debug_dict_score = 0.0

            self.config_data = self.load_config_from_file("myconfig.json")
            do_sync = self.config_data["do_sync"] # default NO
            if do_sync == 1:
                self.log_with_level("Do", 1)
            else:
                self.log_with_level("Not Do", WARNING_LEVEL)

            while do_sync == 1 and count < 1 and len(eval_uids) > 0:
                loop_count += 1
                if loop_count % 20 == 0:
                    tplr.logger.info(f"Loop count is {loop_count}")
                for uid in eval_uids:
                    result = await self.comms.get(
                        uid=str(uid),
                        window=step_window,
                        key="debug",
                        local=False,
                        stale_retention=10,
                    )
                    if not result.success:
                        continue
                    else:
                        result = cast(dict, result.data)
                        debug_dict = result
                        print(f"uid is {uid}")
                        debug_dict_score = sync_scores[uid]
                        count += 1
                    if count >= 1:
                        break
                if loop_count > 200:
                    break

            # check if my debug dict is exist
            my_config = self.load_config_from_file("myconfig.json")
            conv = {"iia":248, "iib":213, "iic":55, "iid":212, "iie":227, "iif":69}
            my_uid = conv[my_config["wallet.hotkey"]]
            result = await self.comms.get(
                uid=str(my_uid),
                window=step_window,
                key="debug",
                local=False,
                stale_retention=10,
            )
            if result.success:
                result = cast(dict, result.data)
                success = True
            else:
                success = False

            if not success:
                config_data = self.load_config_from_file("myconfig.json")
                submit_uids = config_data["sync_uid"]
                for uid in submit_uids:
                    await self.comms.put(
                        state_dict=debug_dict,
                        uid=str(uid),
                        window=step_window,
                        key="debug",
                        local=False,
                        stale_retention=100,
                    )
                    tplr.logger.info(f"{tplr.T()} Submitting debug dict for UID {uid}")
            else:
                tplr.logger.info(f"Debug dict is None. Failed to get debug dict")


    async def wandb_sync(self):
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                self.wandb_info.sync()
                tplr.logger.info("WandB sync complete")
                break
            except Exception as e:
                if "timeout" in str(e).lower() or "read timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        tplr.logger.warning(f"WandB sync timeout (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # exponential backoff
                    else:
                        tplr.logger.error(f"WandB sync failed after {max_retries} attempts: {e}")
                else:
                    tplr.logger.error(f"WandB sync error: {e}")
                    break
        
        await asyncio.sleep(0)


def load_config_from_file(file_path: str):
    config_data = {}
    with open(file_path, "r") as f:
        config = json.load(f)
        config_data.update(config)
    return config_data

# Start simple miner.
if __name__ == "__main__":
    uvloop.install()
    try:
        asyncio.run(Fake().main())
    except KeyboardInterrupt:
        pass
