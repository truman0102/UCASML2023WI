# UCASML2023WI
2023冬国科大模式识别与机器学习期末作业

The project is organized as follows:

```bash
UCASML2023WI
├── README.md
├── data
│   ├── README.md
│   └── cifar-10-batches-py
│       ├── readme.html
│       ├── batches.meta
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       └── test_batch
├── model
│   ├── README.md
│   ├── ResNet
│   │   ├── README.md
│   │   ├── resnet50-19c8e357.pth
│   │   ├── checkpoint.pth
│   │   └── ...
│   └── ...
├── code
│   ├── README.md
│   ├── data.ipynb
│   ├── template.ipynb
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── utils.py
│   └── ...
├── README.md
├── papers.bib
├── requirements.txt
└── ...
```

You can refer to the [template](/code/template.ipynb) to write your own code

## Model

### Linear Model

### Non-Linear Model

### Ensemble Model

### ResNet

Introduced in pytorch hub [tutorial](https://pytorch.org/hub/pytorch_vision_resnet/).

See source code [here](https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html).

### ~~Vision Transformer (ViT)~~

It was introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

The pytorch version of ViT in this project is from [huggingface](https://huggingface.co/google/vit-base-patch16-224).