# Self-Supervised Vertical Federated Learning

Code for running SS-VFL-I and SS-VFL-C.
This directory builds off of the [Momentum Contrast (MoCo) with Alignment and Uniformity Losses repo](https://github.com/SsnL/moco_align_uniform) to simulate a vertical federated learning setting.
More details on the algorithm can be found in our paper: [**Self-Supervised Vertical Federated Learning**](https://openreview.net/pdf?id=z2RNsvYZZTf):

```
@inproceedings{castiglia2022ssvfl,
  title={Self-Supervised Vertical Federated Learning},
  author={Castiglia, Timothy and Das, Anirban and Wang, Shiqiang and Patterson, Stacy},
  booktitle={Workshop on Federated Learning: Recent Advances and New Challenges},
  year={2022}
}
```

## Running SS-VFL with the ModelNet10 dataset and ImageNet dataset

### Dependencies
One can install our environment with Anaconda:
```bash
conda env create -f flearn.yml 
```

### Datasets

For ModelNet10, download ModelNet40 from the following link and create a folder under "data/10class/classes" that contains just the classes: bathtub, bed, chair, desk, dresser, monitor, night stand, sofa, table, toilet: [Google Drive](https://drive.google.com/file/d/1YaGWesl9DyYNoE8Pfe80EmqHkoJ0XlKU/view?usp=sharing).

The ImageNet dataset can be downloaded from [image-net.org](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
The data must be placed in a folder named 'imagenet', then preprocessed with:
```.bash
python create_imagenet_subset.py imagenet data/imagenet100 
```

### Running code

#### ImageNet

To run contrastive learning pre-training for a party with the ImageNet dataset:
```.bash
python main_moco_VFL.py -a resnet50 --moco-contr-w 1.0 --moco-contr-tau 0.2 --moco-unif-w 0 --moco-unif-t 0 --moco-align-w 0 --moco-align-alpha 0 --lr 0.03 --batch-size 128 --world-size 1 --rank 0 -j 16 --multiprocessing-distributed --client $client ./data/imagenet100/
```
where $client is the client ID. One must run this code for both client 0 and 1 before running downstream training.
Then, to run downstream supervised training:
```.bash
python main_VFLcls.py -a resnet50 --lr 0.03 --schedule 150 300 --batch-size 256 --mode $mode --labeled_frac $frac --epochs 500 --world-size 1 --rank 0 -j 16 --multiprocessing-distributed --pretrained ./results/mocom_VFL0_0.999_contr1tau0.2_b128_lr0.03_e120,160,200/checkpoint_0199.pth.tar 
```
where $mode is set to 'supervised', 'unsupervised, or 'semi' depending on if you are running Supervised VFL, SS-VFL-I, or SS-VFL-C, respectively,
and $frac is the fraction of the full dataset that is labeled (for example, frac=0.1 means that 10% of the dataset is labeled).

Loss and accuracy results are generated as pickles files in the current working directory.

#### ModelNet

To run contrastive learning pre-training for a party with the ModelNet dataset:
```.bash
python main_moco_mvcnn.py -a resnet18 --moco-contr-w 1.0 --moco-contr-tau 0.2 --moco-unif-w 0 --moco-unif-t 0 --moco-align-w 0 --moco-align-alpha 0 --lr 0.03 --batch-size 32 --world-size 1 --rank 0 -j 16 --multiprocessing-distributed --client $client ./data/10class/classes/
```
where $client is the client ID. One must run this code for clients 0-11 before running downstream training.
Then, to run downstream supervised training:
```.bash
python main_MVCNNcls.py -a resnet18 --seed $seed --lr 0.01 --schedule 150 300 --batch-size 64 --mode $mode --labeled_frac $frac --epochs 500 --world-size 1 --rank 0 -j 16 --multiprocessing-distributed --pretrained ./results/mocom_mvcnn0_0.999_contr1tau0.2_b32_lr0.03_e120,160,200/checkpoint_0199.pth.tar ./data/10class/classes/
```
where $mode is set to 'supervised', 'unsupervised, or 'semi' depending on if you are running Supervised VFL, SS-VFL-I, or SS-VFL-C, respectively,
and $frac is the fraction of the full dataset that is labeled (for example, frac=0.1 means that 10% of the dataset is labeled).

Loss and accuracy results are generated as pickles files in the current working directory.

#### Experiments in paper
To run the experiments from our paper sequentially:
```.bash
python run_unsupervised.py
python run_downstream.py
```

To generate plots from our paper:
```.bash
python plot_semi.py
```

## Additional Citations
If you use this code in your research, please cite the original authors of the moco_align_uniform repo: [**Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere**](https://arxiv.org/abs/2005.10242):
```
@inproceedings{wang2020hypersphere,
  title={Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
  author={Wang, Tongzhou and Isola, Phillip},
  booktitle={International Conference on Machine Learning},
  organization={PMLR},
  pages={9929--9939},
  year={2020}
}
```

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
