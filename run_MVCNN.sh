#!/bin/bash -x

seed=$1
frac=$2
mode=$3
port=$4

python main_MVCNNcls.py -a resnet18 --seed $seed --lr 0.01 --schedule 150 300 --batch-size 64 --mode $mode --labeled_frac $frac --epochs 500 --world-size 1 --rank 0 -j 16 --multiprocessing-distributed --pretrained ./results/mocom_mvcnn0_0.999_contr1tau0.2_b32_lr0.03_e120,160,200/checkpoint_0199.pth.tar ./data/10class/classes/ --dist-url tcp://localhost:$port
