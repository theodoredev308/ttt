deactivate
mv checkpoints ../sec
mv tokenized ../sec
cd ../sec
source .venv/bin/activate
pm2 delete 0
pm2 start run.sh
pm2 log 0
