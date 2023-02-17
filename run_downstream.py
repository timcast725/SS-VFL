import numpy as np
import os
import random

fracs = ['0.01', '0.05', '0.1', '0.25']
port = 10001
seeds = [707412115,1928644128,16910772,1263880818,1445547577]
num_clients = 2
for seed in seeds:
    for frac in fracs:
        os.system(f'./run_VFL.sh {seed} {frac} supervised {port}')
        os.system(f'./run_VFL.sh {seed} {frac} unsupervised {port+1}')
        os.system(f'./run_VFL_resume.sh {seed} {frac} semi {port+2}')
        port += 3

## ModelNet
num_clients = 12
for seed in seeds:
    for frac in fracs:
        os.system(f'./run_MVCNN.sh {seed} {frac} supervised {port}')
        os.system(f'./run_MVCNN.sh {seed} {frac} unsupervised {port+1}')
        os.system(f'./run_MVCNN.sh {seed} {frac} semi {port+2}')
        port += 3
