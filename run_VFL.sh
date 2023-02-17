#!/bin/bash -x

seed=$1
frac=$2
mode=$3
port=$4

python main_VFLcls.py -a resnet50 --seed $seed --lr 0.03 --schedule 150 300 --batch-size 256 --mode $mode --labeled_frac $frac --epochs 500 --world-size 1 --rank 0 -j 16 --multiprocessing-distributed --pretrained ./results/mocom_VFL0_0.999_contr1tau0.2_b128_lr0.03_e120,160,200/checkpoint_0199.pth.tar ./data/imagenet100/ --dist-url tcp://localhost:$port
