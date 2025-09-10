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

"""
Lightweight Checkpoint Downloader

This script downloads checkpoints from validators' R2 buckets without using GPU/CPU resources.
It extracts the essential checkpoint download logic from the miner without the training components.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import bittensor as bt
import uvloop

import tplr
from tplr.dcp_checkpoint import DCPCheckpointer


class LightweightCheckpointDownloader:
    """Lightweight checkpoint downloader that only downloads checkpoints from R2 buckets."""
    
    def __init__(self):
        self.config = self._load_config()
        self.wallet = bt.wallet(config=self.config)
        self.comms = None
        self.uid = None
        self.ckpt = None
        
    def _load_config(self):
        """Load configuration from command line arguments."""
        parser = argparse.ArgumentParser(description="Lightweight checkpoint downloader")
        parser.add_argument(
            "--netuid", type=int, default=3, help="Bittensor network UID."
        )
        parser.add_argument(
            "--device", type=str, default="cpu", help="Device to use (cpu only for this script)"
        )
        parser.add_argument(
            "--debug", action="store_true", help="Enable debug logging"
        )
        parser.add_argument(
            "--checkpoint-version", 
            type=str, 
            default=None,
            help="Specific checkpoint version to download (default: current version)"
        )
        parser.add_argument(
            "--checkpoint-window",
            type=int,
            default=None,
            help="Specific window to download (default: latest)"
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="./checkpoints",
            help="Directory to save downloaded checkpoints"
        )
        parser.add_argument(
            "--prefer-highest-staked",
            action="store_true",
            default=True,
            help="Prefer downloading from highest-staked validators"
        )
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        
        config = bt.config(parser)
        if config.debug:
            tplr.debug()
            
        return config
    
    def load_config_from_file(self, file_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(file_path, "r") as f:
                config_data = json.load(f)
            return config_data
        except FileNotFoundError:
            tplr.logger.error(f"Config file not found at {file_path}")
            raise
        except Exception as e:
            tplr.logger.error(f"Error loading {file_path}: {e}")
            raise
    
    async def initialize_comms(self):
        """Initialize communication components without GPU/CPU intensive operations."""
        tplr.logger.info("Initializing communication components...")
        
        # Load config from file if it exists
        try:
            config_data = self.load_config_from_file("myconfig.json")
            self.config.wallet.name = config_data["wallet.name"]
            self.config.wallet.hotkey = config_data["wallet.hotkey"]
        except Exception as e:
            tplr.logger.warning(f"Could not load myconfig.json: {e}")
        
        # Initialize comms (this handles R2 bucket access)
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=self.config,
            hparams=tplr.load_hparams(use_local_run_hparams=False),
            uid=None,  # Will be set after comms initialization
        )
        
        # Get UID from metagraph
        if self.wallet.hotkey.ss58_address not in self.comms.metagraph.hotkeys:
            tplr.logger.error(
                f"Wallet {self.wallet} is not registered on subnet: {self.comms.metagraph.netuid}"
            )
            sys.exit(1)
        
        self.uid = self.comms.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.comms.uid = self.uid
        self.comms.uid2 = self.uid  # Use same UID for simplicity
        
        tplr.logger.info(f"Initialized with UID: {self.uid}")
        
        # CRITICAL: Load commitments to access validator buckets
        tplr.logger.info("Loading validator commitments...")
        self.comms.commitments = await self.comms.get_commitments()
        tplr.logger.info("Loaded commitments")
        
        # Initialize checkpoint manager
        version = self.config.checkpoint_version or tplr.__version__
        self.ckpt = DCPCheckpointer(
            self.comms, 
            uid=self.uid, 
            version=version, 
            repo_root=self.config.output_dir
        )
        
        tplr.logger.info(f"Checkpoint manager initialized for version: {version}")
    
    async def discover_latest_checkpoint(self) -> Optional[int]:
        """Discover the latest available checkpoint window."""
        tplr.logger.info("Discovering latest checkpoint...")
        
        try:
            # First try the current version
            latest_window = await self.ckpt._discover_latest(
                prefer_highest_staked=self.config.prefer_highest_staked
            )
            
            if latest_window is not None:
                tplr.logger.info(f"Found latest checkpoint at window: {latest_window}")
                return latest_window
            
            # If no checkpoints found in current version, try to find any available versions
            tplr.logger.info("No checkpoints found in current version, checking for other versions...")
            
            # Get the bucket for listing
            bucket = await self.ckpt._choose_read_bucket(
                prefer_highest_staked=self.config.prefer_highest_staked
            )
            s3 = await self.comms._get_s3_client(bucket)
            
            # List all checkpoints to find available versions
            response = await s3.list_objects_v2(Bucket=bucket.name, Prefix="checkpoints/")
            
            if 'Contents' in response:
                versions = set()
                for obj in response['Contents']:
                    key = obj['Key']
                    parts = key.split('/')
                    if len(parts) >= 2:
                        versions.add(parts[1])
                
                if versions:
                    tplr.logger.info(f"Found available versions: {sorted(versions)}")
                    # Try the latest version found
                    latest_version = sorted(versions)[-1]
                    tplr.logger.info(f"Trying version: {latest_version}")
                    
                    # Create a new checkpoint manager with the found version
                    test_ckpt = DCPCheckpointer(
                        self.comms, 
                        uid=self.uid, 
                        version=latest_version, 
                        repo_root=self.config.output_dir
                    )
                    
                    latest_window = await test_ckpt._discover_latest(
                        prefer_highest_staked=self.config.prefer_highest_staked
                    )
                    
                    if latest_window is not None:
                        tplr.logger.info(f"Found checkpoint in version {latest_version} at window: {latest_window}")
                        # Update our checkpoint manager to use the found version
                        self.ckpt = test_ckpt
                        return latest_window
            
            tplr.logger.warning("No checkpoints found in any version")
            return None
            
        except Exception as e:
            tplr.logger.error(f"Error discovering latest checkpoint: {e}")
            return None
    
    async def download_checkpoint(self, window: Optional[int] = None) -> bool:
        """Download checkpoint for the specified window."""
        if window is None:
            window = await self.discover_latest_checkpoint()
            if window is None:
                tplr.logger.error("No checkpoint window available for download")
                return False
        
        tplr.logger.info(f"Downloading checkpoint for window: {window}")
        
        try:
            # Download all checkpoint files
            local_dir = await self.ckpt.download_all(
                window=window,
                prefer_highest_staked=self.config.prefer_highest_staked
            )
            
            if local_dir is not None:
                tplr.logger.info(f"Successfully downloaded checkpoint to: {local_dir}")
                return True
            else:
                tplr.logger.error("Failed to download checkpoint")
                return False
                
        except Exception as e:
            tplr.logger.error(f"Error downloading checkpoint: {e}")
            return False
    
    async def list_available_checkpoints(self):
        """List available checkpoints without downloading them."""
        tplr.logger.info("Listing available checkpoints...")
        
        try:
            # Get the bucket for listing
            bucket = await self.ckpt._choose_read_bucket(
                prefer_highest_staked=self.config.prefer_highest_staked
            )
            
            # Check if this is a validator bucket or our own bucket
            if hasattr(self.comms, 'commitments') and self.comms.commitments:
                # Try to get highest-staked validator bucket for comparison
                try:
                    validator_bucket, validator_uid = await self.comms._get_highest_stake_validator_bucket()
                    if validator_bucket:
                        print(f"Highest-staked validator UID: {validator_uid}")
                        print(f"Validator bucket: {validator_bucket}")
                        if bucket.name == validator_bucket.name:
                            print("✅ Using validator bucket (correct!)")
                        else:
                            print("❌ Using own bucket instead of validator bucket")
                    else:
                        print("❌ Could not get validator bucket")
                except Exception as e:
                    print(f"❌ Error getting validator bucket: {e}")
            else:
                print("❌ No commitments loaded - cannot access validator buckets")
            
            s3 = await self.comms._get_s3_client(bucket)
            
            # First, let's see what's actually in the bucket
            tplr.logger.info("Checking what's in the bucket...")
            response = await s3.list_objects_v2(Bucket=bucket.name, Prefix="checkpoints/")
            
            # List checkpoints for current version
            prefix = f"checkpoints/{self.ckpt.version}/"
            tplr.logger.info(f"Looking for checkpoints with prefix: {prefix}")
            response = await s3.list_objects_v2(Bucket=bucket.name, Prefix=prefix)

            if 'Contents' in response and response['Contents']:
                windows = set()
                versions = set()
                for obj in response['Contents']:
                    key = obj['Key']
                    # Extract version and window from path like "checkpoints/version/window/file"
                    parts = key.split('/')
                    if len(parts) >= 2:
                        versions.add(parts[1])  # Add version
                    if len(parts) >= 3:
                        try:
                            window = int(parts[2])
                            windows.add(window)
                        except ValueError:
                            continue
                
                tplr.logger.info(f"Available versions: {sorted(versions)}")
                
                if windows:
                    sorted_windows = sorted(windows, reverse=True)
                    tplr.logger.info(f"Available checkpoint windows: {sorted_windows}")
                    tplr.logger.info(f"Latest window: {max(windows)}")
                    return sorted_windows
                else:
                    tplr.logger.warning("No checkpoint windows found")
                    return []
            else:
                tplr.logger.warning("❌ No checkpoints found in bucket")
                tplr.logger.info("This could mean:")
                tplr.logger.info("  1. Validators haven't uploaded any checkpoints yet")
                tplr.logger.info("  2. Checkpoints are stored in a different bucket")
                tplr.logger.info("  3. Checkpoints use a different version format")
                tplr.logger.info("  4. The network is still in warmup phase")
                return []
                
        except Exception as e:
            tplr.logger.error(f"Error listing checkpoints: {e}")
            return []
    
    async def check_alternative_locations(self):
        """Check for checkpoints in alternative locations or formats."""
        tplr.logger.info("Checking alternative checkpoint locations...")
        
        try:
            # Get bucket
            bucket = await self.ckpt._choose_read_bucket(
                prefer_highest_staked=self.config.prefer_highest_staked
            )
            s3 = await self.comms._get_s3_client(bucket)
            
            # Check different possible prefixes
            prefixes_to_check = [
                "checkpoints/",
                "gradients/",
                "models/",
                "state/",
                "snapshots/",
                "backups/",
                "checkpoint/",  # singular
                "model/",       # singular
            ]
            
            for prefix in prefixes_to_check:
                tplr.logger.info(f"Checking prefix: {prefix}")
                response = await s3.list_objects_v2(Bucket=bucket.name, Prefix=prefix, MaxKeys=10)
                
                if 'Contents' in response and response['Contents']:
                    tplr.logger.info(f"✅ Found {len(response['Contents'])} objects under '{prefix}':")
                    for obj in response['Contents'][:5]:  # Show first 5
                        tplr.logger.info(f"  - {obj['Key']}")
                    if len(response['Contents']) > 5:
                        tplr.logger.info(f"  ... and {len(response['Contents']) - 5} more")
                else:
                    tplr.logger.info(f"❌ No objects found under '{prefix}'")
            
            # Also check if there are any objects at all in the bucket
            tplr.logger.info("Checking if bucket has any objects at all...")
            response = await s3.list_objects_v2(Bucket=bucket.name, MaxKeys=10)
            
            if 'Contents' in response and response['Contents']:
                tplr.logger.info(f"✅ Bucket contains {len(response['Contents'])} objects:")
                for obj in response['Contents']:
                    tplr.logger.info(f"  - {obj['Key']}")
                
                # Analyze what types of objects we have
                debug_files = [obj for obj in response['Contents'] if 'debug-' in obj['Key']]
                checkpoint_files = [obj for obj in response['Contents'] if 'checkpoint' in obj['Key'].lower()]
                gradient_files = [obj for obj in response['Contents'] if 'gradient' in obj['Key'].lower()]
                
                tplr.logger.info(f"  - Debug files: {len(debug_files)}")
                tplr.logger.info(f"  - Checkpoint files: {len(checkpoint_files)}")
                tplr.logger.info(f"  - Gradient files: {len(gradient_files)}")
                
                # Extract version from debug files
                if debug_files:
                    versions = set()
                    for obj in debug_files:
                        key = obj['Key']
                        if '-v' in key:
                            version_part = key.split('-v')[1].split('.pt')[0]
                            versions.add(version_part)
                    if versions:
                        tplr.logger.info(f"  - Available versions in debug files: {sorted(versions)}")
                        
                        # Try to find checkpoints in the version that has debug files
                        for version in sorted(versions, reverse=True):
                            tplr.logger.info(f"  - Checking for checkpoints in version {version}...")
                            checkpoint_prefix = f"checkpoints/{version}/"
                            ckpt_response = await s3.list_objects_v2(Bucket=bucket.name, Prefix=checkpoint_prefix, MaxKeys=5)
                            if 'Contents' in ckpt_response and ckpt_response['Contents']:
                                tplr.logger.info(f"    ✅ Found {len(ckpt_response['Contents'])} checkpoint objects in version {version}")
                                for ckpt_obj in ckpt_response['Contents']:
                                    tplr.logger.info(f"      - {ckpt_obj['Key']}")
                            else:
                                tplr.logger.info(f"    ❌ No checkpoints found in version {version}")
            else:
                tplr.logger.warning("❌ Bucket appears to be completely empty")
                
        except Exception as e:
            tplr.logger.error(f"Error checking alternative locations: {e}")

    def _local_checkpoints_exist(self, window: int = None) -> bool:
        """
        Check if checkpoint files already exist locally for the given window.
        If window is None, check for any window in the output directory.
        """
        output_dir = Path(self.config.output_dir)
        version = self.ckpt.version if self.ckpt else (self.config.checkpoint_version or tplr.__version__)
        checkpoints_dir = output_dir / "checkpoints" / str(version)
        if not checkpoints_dir.exists():
            return False
        if window is not None:
            window_dir = checkpoints_dir / str(window)
            if window_dir.exists() and any(window_dir.iterdir()):
                tplr.logger.info(f"Checkpoint for window {window} already exists locally at {window_dir}")
                return True
            return False
        # If window is None, check for any window directory with files
        for subdir in checkpoints_dir.iterdir():
            if subdir.is_dir() and any(subdir.iterdir()):
                tplr.logger.info(f"Checkpoint already exists locally at {subdir}")
                return True
        return False

    async def run(self):
        """Main execution function."""
        tplr.logger.info("Starting lightweight checkpoint downloader...")
        
        try:
            # Initialize communication components
            await self.initialize_comms()
            
            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # List available checkpoints
            available_windows = await self.list_available_checkpoints()
            
            if not available_windows:
                tplr.logger.warning("No checkpoints found in standard location")
                await self.check_alternative_locations()
                tplr.logger.error("No checkpoints available for download")
                return False
            
            # Download specified window or latest
            target_window = self.config.checkpoint_window
            if target_window is None:
                target_window = max(available_windows)
                tplr.logger.info(f"Using latest window: {target_window}")
            elif target_window not in available_windows:
                tplr.logger.error(f"Window {target_window} not available. Available: {available_windows}")
                return False

            # Check if checkpoint already exists locally
            if self._local_checkpoints_exist(target_window):
                tplr.logger.info(f"Checkpoint for window {target_window} already exists locally. Skipping download.")
                return True

            # Download the checkpoint
            success = await self.download_checkpoint(target_window)
            
            if success:
                tplr.logger.info("Checkpoint download completed successfully!")
                return True
            else:
                tplr.logger.error("Checkpoint download failed!")
                return False
                
        except Exception as e:
            tplr.logger.error(f"Error in main execution: {e}")
            return False


async def main():
    """Main entry point."""
    downloader = LightweightCheckpointDownloader()
    success = await downloader.run()
    
    if success:
        tplr.logger.info("Checkpoint downloader completed successfully!")
        sys.exit(0)
    else:
        tplr.logger.error("Checkpoint downloader failed!")
        sys.exit(1)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
