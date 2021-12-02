# Self-Supervised Graph Representation Learning via Topology Transformations

This repository is the official PyTorch implementation of the following paper:

Xiang Gao, Wei Hu, Guo-Jun Qi, "Self-Supervised Graph Representation Learning via Topology Transformations," _IEEE Transactions on Knowledge and Data Engineering (TKDE)_, December 2021.

## Requirements

 - Python3>=3.7.10
 - pytorch>=1.9.0
 - tensorboardX>=1.9
 - torch_geometric>=1.7.2

**Note:** We are not sure whether the code can be run properly by using a lower version of the relevant package.

## Usage

**Tip:** Experimental results of graphs usually face greater randomness than images. We suggest you run the experiment more than one time and select the best result.

### Unsupervised Training

To train a feature extractor in an unsupervised fashion, run

```shell
python main.py --phase backbone --dataset cora --perturbation-rate 0.7 --hidden-channels 512 --k 2 --num-epochs 512 --lr 0.0001 --use-cuda true --device 0 --save-dir ./results
```

### Supervised Evaluation

After training the feature extractor, you need to train the classifier by running the following command:

```shell
python main.py --phase classifier --dataset cora --perturbation-rate 0.7 --hidden-channels 512 --k 2 --backbone ./results/cora_best.pt --lr 0.001 --use-cuda true --device 0 --save-dir ./results
```

## Reference

Please cite our paper if you use any part of the code from this repository:

```text
@article{gao2021topoter,
  title={Self-Supervised Graph Representation Learning via Topology Transformations},
  author={Gao, Xiang and Hu, Wei and Qi, Guo-Jun},
  journal={IEEE Transactions on Knowledge and Data Engineering (TKDE)},
  month={December},
  year={2021}
}
```

## Acknowledgement

Our code is released under MIT License (see LICENSE for details). Some of the code in this repository was borrowed from the following repositories:

 - [DGI](https://github.com/PetarV-/DGI)
 - [GMI](https://github.com/zpeng27/GMI)
