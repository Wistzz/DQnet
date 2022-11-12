#!/usr/bin/env bash
set -e         
set -u      
set -x          
set -o pipefail 

export CUDA_VISIBLE_DEVICES="$1"
echo 'Excute the script on GPU: ' "$1"

echo 'For COD'
python test.py --config ./configs/DQnet/DQnet.py \
    --model-name DQnet \
    --batch-size 22 \
    --load-from  \
    --save-path 

echo 'For SOD'
python test.py --config ./configs/DQnet/DQnet.py \
    --model-name DQnet \
    --batch-size 22 \
    --load-from  \
    --save-path 
