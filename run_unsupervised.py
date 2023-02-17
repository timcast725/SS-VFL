import numpy as np
import os
import random

## ImageNet
port = 10001
seeds = [707412115,1928644128,16910772,1263880818,1445547577]
num_clients = 2
for seed in seeds:
    for client in range(num_clients):
        os.system(f'./run_contr.sh {seed} {client} {port}')
        port += 1

## ModelNet
num_clients = 12
for seed in seeds:
    for client in range(num_clients):
        os.system(f'./run_slurm_mvcnn.sh {seed} {client} {port}')
        port += 1

