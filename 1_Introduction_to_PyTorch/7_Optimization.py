#
# 2022.06.11
#
# Created by 김성찬.
#

#
# Optimizing Model Parameters
#

# 모델 학습은 반복적인 과정이다;
# epoch라고 불리는, 각각의 iteration에서 모델은 결과를 예측하고, 예측에 대한 오류를 계산한다.
# 각각의 파라미터에 대해 오류에 대한 derivative를 모으고
# gradient descent를 사용하여 이 파라미터들을 최적화한다.

# Prerequisite Code

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# Hyperparameters

# Hyperparameters는 조절할 수 있는 파라미터로 모델 최적화 과정을 조절할 수 있도록 한다.
# hyperparameter 값은 모델 학습과 수렴 속도에 영향을 줄 수 있다.

# Number of Epochs: 데이터셋에 대해 반복할 횟수
# Batch Size: 파라미터를 업데이트하기 전에 네트워크를 통과시킬 데이터 샘플의 수
# Learning Rate: 각각의 batch/epoch에 대해 모델 파라미터를 얼마나 업데이트할 지에 대한 값.
#                낮은 값은 학습 속도를 느리게 하고, 높은 값은 학습 동안 예측할 수 없는 양상을 초래할 수 있다.

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Optimization Loop

# 최적화 루프에서 한 번 반복하는 것을 epoch라고 부른다.
# 각각의 에포크는 2개의 부분으로 구성된다.
# - 학습 루프: 학습 데이터셋에 대해 반복하고 최적의 파라미터로 수렴하도록 한다.
# - 검증 루프: 검증 데이터셋에 대해 반복하여 모델 성능이 향상되고 있는 지 확인한다.

# Loss Function

# 학습되지 않은 네트워크는 주어진 학습 데이터에 대해 정답을 반환하지 못할 수도 있다.
# Loss Function은 네트워크의 결과와 정답 사이의 불일치 정도를 측정한다.
# 우리는 학습 과정에서 loss function을 최소화해야 한다.
# loss를 계산하기 위해 주어진 데이터 샘플 입력에 대한 예측을 만들고 정답 데이터 레이블 값과 비교한다.

# 일반적인 loss function으로 회귀를 위한 nn.MSELoss (Mean Square Error)와 
# 분류를 위한 nn.NLLLoss (Negative Log Likelihood)가 있다.
# nn.CrossEntropyLoss는 nn.LogSoftmax와 nn.NLLLoss를 결합한 것이다.

# 모델의 출력 logits를 nn.CrossEntropyLoss에 전달한다.
# 이것은 logits를 normalize하고 예측의 오류를 계산한다.

loss_fn = nn.CrossEntropyLoss()

# Optimizer

# 최적화는 학습 단계에서 모델 에러를 줄이기 위해 모델 파라미터를 업데이트하는 과정이다.
# 최적화 알고리즘은 이 과정이 어떻게 동작하는 지를 정의한다.
# 모든 최적화 알고리즘은 optimizer object에 캡슐화되어 있다.

# 학습되어야 하는 모델 파라미터와 learning rate hyperparameter를 전달하여 optimizer를 초기화한다.

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 학습 루프 내에서, 최적화는 세 단계를 걸쳐 진행된다.
# - optimizer.zero_grad()를 실행하여 모델 파라미터의 gradient를 리셋한다.
#   기본적으로 gradient는 더해진다. 따라서 double-counting을 방지하기 위해 각각의 루프에서 0으로 만들어야 한다.
# - loss.backward()를 실행하여 예측에 대한 loss를 backpropagate한다.
# - gradient를 계산한 이후, optimizer.step()을 사용하여 
#   backward pass에서 얻은 gradient를 통해 파라미터를 업데이트한다.

# Full Implementation

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# loss function과 optimizer을 초기화하여 train_loop와 test_loop에 전달한다.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
