#!/bin/bash -x

seed=$1

python main_lincls.py -a resnet50 --lr 30.0 --batch-size 256 --pretrained ./results/mocom0.999_contr1tau0.2_b128_lr0.03_e120,160,200/checkpoint_0199.pth.tar --world-size 1 --rank 0 -j 1 --epochs 100 --multiprocessing-distributed ./data/imagenet100/
