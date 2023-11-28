##!/bin/usr/env bash

SEED=196
PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/vqa.py --train train --valid val  --llayers 9 --xlayers 5 --rlayers 5 --batchSize 32 --optim bert --lr 5e-5 --epochs 50 \
--tqdm --name vqa-cp-test --output /nas-hdd/tarbucket/adyasha/models/vqa-cp/vqa-cp-causal-0.25-contrastive-no-norm-finetuned-lr-5e-5-seed-${SEED}/ \
--seed ${SEED} --loss-fn Farm --use-farm --farm-coeff 0.25 --gpu 1 --causal-model --dynamic-coeff --wandb --contrastive \
--load ./output/pretrained/vqa-cp_lxrt_pretrained.pth