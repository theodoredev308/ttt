torchrun --nproc_per_node=8 neurons/miner.py \
    --actual-batch-size 208 \
    --wallet.name izo \
    --wallet.hotkey iia \
    --device cuda \
    --netuid 3 \
    --subtensor.network finney \
    --sync_state