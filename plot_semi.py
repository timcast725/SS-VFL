"""
Plot adaptive experimental results
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import os
import glob
import math

font = {'family' : 'DejaVu Sans',
#        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
colors=['#6aa2fc', '#fc8181', '#a5ff9e', '#3639ff', '#ff3636', '#13ba00', '#ff62f3']

def all_seeds(dataset, frac, mode, prefix):
    print(prefix)
    if dataset == "VFL":
        files = glob.glob(f'/hdd/home/results/mocom_VFL0_0.999_contr1tau0.2_b128_lr0.03_e120,160,200/VFLcls/checkpoint_0162.pth.tar/{prefix}_seed*/test_acc5.pkl')
    else:
        files = glob.glob(f'/hdd/home/results/mocom_mvcnn0_0.999_contr1tau0.2_b32_lr0.03_e120,160,200/MVCNNcls/checkpoint_0199.pth.tar/{prefix}_seed*/test_acc1.pkl')

    pickles = []    
    min_len = 9999
    for f in files:
        pkl = pickle.load(open(f, 'rb'))
        #pkl = np.array([x.cpu().numpy() for x in pkl])
        if len(pkl) < 400:
            continue
        min_len = min(min_len, len(pkl))
        pickles.append(np.array(pkl))
    pickles = np.array(pickles)
    for i in range(len(pickles)):
        pickles[i] = pickles[i][:min_len]
    pickles = np.array(pickles)

    # How much communication to reach target accuracy?
    bits_total = []
    for i in range(len(pickles)):
        accs = pickles[i]
        final_epoch = 0
        bits_so_far = 0

        if mode == "unsupervised":
            bits_per_epoch = 0

        if dataset == "VFL":
            if frac == '0.01':
                target_acc = 70 
            elif frac == '0.05':
                target_acc = 75 
            elif frac == '0.1':
                target_acc = 80
            elif frac == '0.25':
                target_acc = 85 

            # num_client*embedding_size*num_datapoints*size_of_float
            if mode == "unsupervised":
                bits_so_far = 2*128*126788*float(frac)*32
            else:
                bits_per_epoch = 2*128*126788*float(frac)*32
        else:
            if frac == '0.01':
                target_acc = 65 
            elif frac == '0.05':
                target_acc = 70 
            elif frac == '0.1':
                target_acc = 75
            elif frac == '0.25':
                target_acc = 80 

            # num_client*embedding_size*num_datapoints*size_of_float
            if mode == "unsupervised":
                bits_so_far = 12*128*4000*float(frac)*32
            else:
                bits_per_epoch = 12*128*4000*float(frac)*32 

        while accs[final_epoch] < target_acc:
            bits_so_far += bits_per_epoch
            final_epoch += 1
            if final_epoch >= len(accs):
                break
        if final_epoch >= len(accs):
            bits_so_far = float('inf')
        bits_total.append(bits_so_far)

    bits_avg = np.average(bits_total)
    bits_std = np.std(bits_total)
    print(f"{bits_avg/(8*2**20):.2f} $\pm$ {bits_std/(8*2**20):.2f}")
    avg = np.average(pickles, axis=0)
    std = np.std(pickles, axis=0)

    return (avg, std)

#types = ['train_loss', 'train_acc1', 'train_acc5', 'test_loss', 'test_acc1', 'test_acc5']
types = ['test_acc1', 'test_acc5']
datasets = ['VFL','MVCNN']
for dataset in datasets:
    #for t in types:
    fracs = ['0.01','0.05','0.1', '0.25']
    #fracs = ['0.25']
    if dataset == "VFL":
        t = 'test_acc5'
    else:
        t = 'test_acc1'
    
    for frac in fracs:
        # Parse results
        if dataset == "VFL":
            losses0, std0 = all_seeds(dataset, frac, "supervised",   f'b256_lr0.03_modesupervised_frac{frac}_e150,300,500')
            losses1, std1 = all_seeds(dataset, frac, "unsupervised", f'b256_lr0.03_modeunsupervised_frac{frac}_e150,300,500')
            losses2, std2 = all_seeds(dataset, frac, "semi",         f'b256_lr0.03_modesemi_frac{frac}_e150,300,500')
        else:
            losses0, std0 = all_seeds(dataset, frac, "supervised",   f'b64_lr0.01_modesupervised_frac{frac}_e150,300,500')
            losses1, std1 = all_seeds(dataset, frac, "unsupervised", f'b64_lr0.01_modeunsupervised_frac{frac}_e150,300,500')
            losses2, std2 = all_seeds(dataset, frac, "semi",         f'b64_lr0.01_modesemi_frac{frac}_e150,300,500')
    
        fig, ax = plt.subplots()
        # Plot loss
        plt.plot(losses0, label='VFL')
        plt.plot(losses1, label='SS-VFL-I')
        plt.plot(losses2, label='SS-VFL-C')
      
        plt.fill_between(np.linspace(0,len(losses0)-1,len(losses0)), losses0 - std0, losses0 + std0, alpha=0.3)
        plt.fill_between(np.linspace(0,len(losses1)-1,len(losses1)), losses1 - std1, losses1 + std1, alpha=0.3)
        plt.fill_between(np.linspace(0,len(losses2)-1,len(losses2)), losses2 - std2, losses2 + std2, alpha=0.3)
    
        plt.xlabel('Epochs')
        if t == 'loss':
            #plt.ylim(0, 1)
            plt.ylabel('Loss')
        else:
            plt.ylabel('Accuracy')
    
        plt.legend(loc='lower right')

        #ratio = 0.5
        #xleft, xright = ax.get_xlim()
        #ybottom, ytop = ax.get_ylim()
        #ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
        plt.tight_layout()
        plt.savefig(f'moco{dataset}_{t}_frac{float(frac):.2f}.png')
        #plt.show()
        plt.close()
    
