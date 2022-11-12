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
    --load-from ./output/ForSharing/cod_zoomnet_r50_bs8_e40_2022-03-04.pth \
    --save-path ./output/ForSharing/COD_Results

echo 'For SOD'
python test.py --config ./configs/DQnet/DQnet.py \
    --model-name DQnet \
    --batch-size 22 \
    --load-from ./output/ForSharing/sod_zoomnet_r50_bs22_e50_2022-03-04_fixed.pth \
    --save-path ./output/ForSharing/SOD_Results
