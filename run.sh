torchrun --nproc_per_node=8 neurons/miner.py \
    --actual-batch-size 208 \
    --wallet.name multisig-jjpes-atel \
    --wallet.hotkey warm \
    --device cuda \
    --netuid 3 \
    --subtensor.network finney \
    --sync_state
