#
# 2022.06.10
#
# Created by 김성찬.
#

#
# Build The Neural Network
#

# Neural networks는 데이터에 대해 연산을 실행하는 층과 모듈로 구성되어 있다.
# torch.nn namespace는 사용자 정의 neural network를 만들기 위한 모든 블록을 제공한다.
# 파이토치의 모든 모듈은 nn.Module의 하위 클래스이다.
# Neural network는 다른 모듈로 구성된 모듈이다. 이 중첩된 구조는 복잡한 아키텍쳐를 쉽게 만들고 관리할 수 있도록 한다.

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Get Device for Training

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define the Class

# nn.Module을 상속 받아서 neural network를 정의한다.
# __init__에서 neural network layers를 초기화한다.
# 모든 nn.Module의 하위 클래스는 입력 데이터에 대한 연산을 forward 메서드에 구현해야 한다.

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# NeuralNetwork의 인스턴스를 만들고, device로 전달하고, 구조를 출력한다.

model = NeuralNetwork().to(device)
print(model)

# 모델을 사용하기 위해서 입력 데이터를 전달해야 한다. model.forward()를 직접적으로 실행하면 안된다.

# 입력에 대한 함수 실행은 각각의 클래스에 대한 raw predicted values를 가지는 10차원 텐서를 반환한다.
# nn.Softmax 모듈 인스턴스를 통해 예측 확률을 얻는다.

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Model Layers

# FashionMNIST model을 분석해보자.

input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten

# nn.Flatten을 통해 2차원 28x28 이미지를 784 pixel values 배열로 변환한다.

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear

# linear layer은 저장된 가중치와 편향을 사용하여 선형 변환을 적용하는 모듈이다.

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU

# Non-linear activations는 모델의 입력과 출력 사이에 복잡한 매핑을 만들어준다.
# 이것들은 nonlinearity를 적용하기 위해 선형 변환 이후에 적용된다.
# 이를 통해 neural networks는 더 다양한 현상을 학습할 수 있다.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential

# nn.Sequential은 모듈들의 정렬된 컨테이너다. 데이터는 모든 모듈을 정의된 것과 같은 순서대로 통과한다.

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax

# 뉴럴 네트워크의 마지막 linear layer는 범위가 [-infty, infty]인 raw value를 가지는 logits를 반환한다.
# nn.Softmax는 logits를 0과 1 사이의 값으로 scale하여 각각의 클래스에 대해 모델이 예측한 확률을 표현한다.
# dim 파라미터는 해당 차원에 따른 값들의 합이 1이어야 하는 차원을 의미한다.

# Model Parameters

# 뉴럴 네트워크 내 많은 층은 파라미터화되어 있다. 이들은 학습 과정에서 최적화될 가중치와 편향과 연관되어 있다.
# nn.Module을 상속하는 것은 모델 객체 내에 정의된 모든 공간을 자동으로 추적하고,
# 모든 파라미터들을 모델의 parameters() 또는 named_parameters() 메서드를 통해 접근할 수 있게 한다.

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
    