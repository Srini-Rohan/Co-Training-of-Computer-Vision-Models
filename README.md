# Co-Training of Computer Vision Models

## Overview

This project explores the co-training approach for enhancing the performance of computer vision models, specifically focusing on ResNet architectures. The technique involves the creation of two ResNet models, each with randomly removed residual layers, referred to as models with stochastic depth. These models are then co-trained together, with one acting as the parent and the other as the child, and vice versa. 

The main objective of this experimental study is to investigate the impact of co-training and stochastic depth on the accuracy of ResNet models. By exploring different combinations of parent and child models with varying stochastic depth, the project aims to provide insights into the effectiveness of this co-training strategy in improving overall model performance.

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/Srini-Rohan/Co-Training-of-Computer-Vision-Models
cd Co-Training-of-Computer-Vision-Models
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Stochastic Depth

Stochastic depth involves randomly removing residual layers during the construction of ResNet models. This introduces a form of dropout at the layer level, enhancing model robustness and preventing overfitting.

### Stochastic Depth Implementation

The implementation involves the introduction of random binary masks, represented by the dictionaries , which dynamically determine whether each residual layer should be  included or skipped during training. These masks are generated  stochastically for each forward pass, introducing variability in the  depth of the model

Ensure that the `Baselayer` class includes the `add_labels` parameter for implementing variable depth.

## Co-Training Process

The co-training process involves training two models simultaneously. One model acts as the parent, providing guidance to the child model, and vice versa. This bidirectional learning helps models benefit from each other's strengths, potentially leading to improved generalisation and accuracy. 

## Results



| Model     | Baseline | Co training |
| --------- | -------- | ----------- |
| ResNet50  | 75.3     | 76.2        |
| ResNet101 | 76.9     | 78.1        |
| ResNet152 | 78.5     | 79.8        |
