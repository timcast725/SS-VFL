# Self-Supervised Vertical Federated Learning

Code for running SS-VFL-I and SS-VFL-C.
This directory builds off of the [Momentum Contrast (MoCo) with Alignment and Uniformity Losses repo](https://github.com/SsnL/moco_align_uniform) to simulate a vertical federated learning setting.
More details on the algorithm can be in our paper: [**Self-Supervised Vertical Federated Learning**](https://openreview.net/pdf?id=z2RNsvYZZTf):

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
To run the experiments sequentially:
```.bash
python run_unsupervised.py
python run_downstream.py
```

To generate plots:
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
