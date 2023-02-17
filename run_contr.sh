#!/bin/bash -x

seed=$1
client=$2
port=$3

python main_moco_VFL.py -a resnet50 --moco-contr-w 1.0 --moco-contr-tau 0.2 --moco-unif-w 0 --moco-unif-t 0 --moco-align-w 0 --moco-align-alpha 0 --lr 0.03 --batch-size 128 --world-size 1 --rank 0 -j 16 --multiprocessing-distributed --client $client ./data/imagenet100/ --dist-url tcp://localhost:$port
