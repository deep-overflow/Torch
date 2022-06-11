#
# 2022.06.10
#
# Created by 김성찬.
#

#
# Transforms
#

# 데이터는 머신러닝 알고리즘을 학습시키기 위해
# 필요한 최종적으로 가공된 형태로 들어오지 않을 수 있다.
# 따라서 데이터를 가공하고 학습에 적합한 형태로 만들기 위해 transforms를 사용한다.

# 모든 TorchVision 데이터셋은 두 개의 파라미터를 가진다.
# transform은 features를 가공하기 위해, target_transform은 레이블을 가공하기 위한 것이다.
# transformation logic을 포함하는 callables를 받는다.

# FashionMNIST features는 PIL 이미지 형식이고 labels는 정수이다.
# 학습을 위해서, features를 normalized tensors로, labels는 one-hot encoded tensors로 변환해야 한다.
# 이러한 변환을 위해서, ToTensor와 Lambda를 사용한다.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# ToTensor()

# ToTensor는 PIL 이미지 또는 넘파이 배열을 FloatTensor로 변환한다.
# 그리고, 이미지의 픽셀 값을 0과 1 사이의 값으로 만든다.

# Lambda Transforms

# Lambda transforms는 사용자 정의 lambda 함수를 적용한다.
# 여기서는 정수를 원-핫 인코딩된 텐서로 변환하는 함수를 정의한다.
# 먼저 크기가 10인 0 텐서를 만든다. 그리고 레이블에 대응하는 인덱스의 값을 1로 변환하는 scatter_을 실행한다.