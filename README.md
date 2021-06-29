# Unbiased Teacher for Semi-Supervised Object Detection

Unbised Teacher for Semi-Supervised Object Detection이라는 논문의 코드를 가져와 커스텀데이터셋에 대하여 학습할 수 있도록 수정한 코드이다.<br>
기 코드에서 수정할 부분이 다소 있었으며, `train_net.py` 파일의 `register()` 부분에서 train, val 부분을 수정한다면 커스텀데이터셋을 통하여 학습할 수 있다.<br>
학습 포맷은 coco format이며, 병렬학습을 지원한다.<br>
단, 아직은 Inference 부분이 구현되어있지 않으나 곧 지원될 것으로 보인다.<br>
<br>
- 실험 결과 각 라벨링 비율 별, 지도학습보다 준지도학습의 성능이 더 높게 나왔으며,<br> 50%의 label을 사용했을 때에는 전체를 사용하여 지도학습 한 결과와 크게 차이가 나지 않았다.(mAP 기준 2)<br>
- 하지만 실험 모델은 Faster R-CNN이며, EfficientDet이나 다른 SOTA모델로 성능 측정을 해보아야 좀 더 효율성을 확인할 수 있을 것으로 기대된다. <br> 본 코드는 Detectron2를 토대로 구축되었으며, Detectron2에서 EfficientDet과 같은 모델을 구현한다면 본 코드에서도 사용이 가능할 것으로 예상된다.<br><br>

* Multi GPU + Custom Dataset 환경 구축
```
vi /workspace/unbiased-teacher/detectron2/data/catalog.py

#custom dataset 정의
def register_() 정의

:q!

vi /workspace/unbiased-teacher/detectron2/data/build.py

#custom dataset 정의 부분 호출
register_()
```

<img src="teaser/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the PyTorch implementation of our paper: <br>
**Unbiased Teacher for Semi-Supervised Object Detection**<br>
[Yen-Cheng Liu](https://ycliu93.github.io/), [Chih-Yao Ma](https://chihyaoma.github.io/), [Zijian He](https://research.fb.com/people/he-zijian/), [Chia-Wen Kuo](https://sites.google.com/view/chiawen-kuo/home), [Kan Chen](https://kanchen.info/), [Peizhao Zhang](https://scholar.google.com/citations?user=eqQQkM4AAAAJ&hl=en), [Bichen Wu](https://scholar.google.com/citations?user=K3QJPdMAAAAJ&hl=en), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/), [Peter Vajda](https://sites.google.com/site/vajdap)<br>
International Conference on Learning Representations (ICLR), 2021 <br>

[[arXiv](https://arxiv.org/abs/2102.09480)] [[OpenReview](https://openreview.net/forum?id=MJIve1zgR_)] [[Project](https://ycliu93.github.io/projects/unbiasedteacher.html)]

<p align="center">
<img src="teaser/figure_teaser.gif" width="85%">
</p>

# Installation

## Prerequisites

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.

## Install PyTorch in Conda env

```shell
# create conda env
conda create -n detectron2 python=3.6
# activate the enviorment
conda activate detectron2
# install PyTorch >=1.5 with GPU
conda install pytorch torchvision -c pytorch
```

## Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

## Dataset download

1. Download COCO dataset

```shell
# download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

2. Organize the dataset as following:

```shell
unbiased_teacher/
└── datasets/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

## Training

- Train the Unbiased Teacher under 1% COCO-supervision

```shell
python train_net.py \
      --num-gpus 8 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16
```

- Train the Unbiased Teacher under 2% COCO-supervision

```shell
python train_net.py \
      --num-gpus 8 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup2_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16
```

- Train the Unbiased Teacher under 5% COCO-supervision

```shell
python train_net.py \
      --num-gpus 8 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup5_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16
```

- Train the Unbiased Teacher under 10% COCO-supervision

```shell
python train_net.py \
      --num-gpus 8 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16
```

## Resume the training

```shell
python train_net.py \
      --resume \
      --num-gpus 8 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 MODEL.WEIGHTS <your weight>.pth
```

## Evaluation

```shell
python train_net.py \
      --eval-only \
      --num-gpus 8 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 MODEL.WEIGHTS <your weight>.pth
```

## Model Weights

For the following results, we use 16 labeled images + 16 unlabeled images on 8 GPUs (single node).

Faster-RCNN:

|   Model      |   Supervision  |             Batch size              |  AP   |  Model Weights |
| :----------: | :------------: | :---------------------------------: | :---: |:-----: |
| R50-FPN      |       1%       | 16 labeled img + 16 unlabeled imgs  | 20.16 | [link](https://drive.google.com/file/d/1NQs5SrQ2-ODEVn_ZdPU_2xv9mxdY6MPq/view?usp=sharing) |
| R50-FPN      |       2%       | 16 labeled img + 16 unlabeled imgs  | 24.16 | [link](https://drive.google.com/file/d/12q-LB4iDvgXGW50Q-bYOahpalUvO3SIa/view?usp=sharing) |
| R50-FPN      |       5%       | 16 labeled img + 16 unlabeled imgs  | 27.84 | [link](https://drive.google.com/file/d/1IJQeRP9wHPU0J27YTea-y3lIW96bMAUu/view?usp=sharing) |
| R50-FPN      |      10%       | 16 labeled img + 16 unlabeled imgs  | 31.39 | [link](https://drive.google.com/file/d/1U9tnJGvzRFSOnOfIHOnelFmlvEfyayha/view?usp=sharing) |

## FAQ

1. Q: Using the lower batch size and fewer GPUs cannot achieve the results presented in the paper?

- A: We train the model with 32 labeled images + 32 unlabeled images per batch for the results presented in the paper, and using the lower batch size leads to lower accuracy. For example, in the 1% COCO-supervision setting, the model trained with 16 labeled images + 16 unlabeled images achieves 19.9 AP as shown in the following table.

|   Experiment GPUs    |         Batch size per node         |             Batch size              |  AP   |
| :------------------: | :---------------------------------: | :---------------------------------: | :---: |
| 8 GPUs/node; 4 nodes |  8 labeled imgs + 8 unlabeled imgs  | 32 labeled img + 32 unlabeled imgs  | 20.75 |
| 8 GPUs/node; 1 node  | 16 labeled imgs + 16 unlabeled imgs | 16 labeled imgs + 16 unlabeled imgs | 20.16 |

## Citing Unbiased Teacher

If you use Unbiased Teacher in your research or wish to refer to the results published in the paper, please use the following BibTeX entry.

```BibTeX
@inproceedings{liu2021unbiased,
    title={Unbiased Teacher for Semi-Supervised Object Detection},
    author={Liu, Yen-Cheng and Ma, Chih-Yao and He, Zijian and Kuo, Chia-Wen and Chen, Kan and Zhang, Peizhao and Wu, Bichen and Kira, Zsolt and Vajda, Peter},
    booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021},
}
```

Also, if you use Detectron2 in your research, please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

## License

This project is licensed under [MIT License](LICENSE), as found in the LICENSE file.
