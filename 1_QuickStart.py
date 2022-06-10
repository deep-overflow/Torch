#
# 2022.06.10
# 
# Created by 김성찬.
#

#
# Start Torch
#

## Working with data

# 파이토치는 데이터에 대한 두 가지 primitive가 있다: torch.utils.data.DataLoader & torch.utils.data.Dataset
# Dataset은 샘플과 그에 대응하는 레이블을 저장한다.
# DataLoader는 Dataset을 iterable로 감싼다.

import enum
from typing_extensions import dataclass_transform
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 파이토치는 데이터셋을 포함하는 domain-specific libraries를 제공한다. ex) TorchText, TorchVision, TorchAudio
# torchvision.datasets는 CIFAR, COCO와 같은 real-world vision data를 위한 Dataset object를 포함한다.

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# root : 데이터셋을 저장할 디렉터리
# train : 학습을 위한 데이터셋인지, 테스트를 위한 데이터셋인지
# download : 데이터셋을 다운로드할 것인지, 아닌지
# transform : Input에 해당하는 데이터를 가공하는 함수
# target_transform : Label에 해당하는 데이터를 가공하는 함수

# Dataset을 DataLoader의 매개변수로 주어서 데이터셋을 iterable로 감싼다.
# 이를 통해 automatic batching, sampling, shuffling, multiprocess data loading을 구현할 수 있다.

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

## Creating Models

# 파이토치에서 neural network를 정의할 때, nn.Module을 상속 받은 class를 만들어야 한다.
# __init__ 함수 내에 네트워크의 층을 정의하고,
# forward 함수 내에 데이터가 어떻게 네트워크를 통과하는 지 구체화한다.
# neural network에서의 연산을 가속하기 위해 이를 GPU로 전달할 수 있다.

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Optimizing the Model Parameters

# 모델을 학습하기 위해서는 Loss 함수와 Optimizer가 필요하다.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 한 번의 학습 루프에서 모델은 학습 데이터셋에 대한 예측을 만들고,
# 예측 에러를 역전파하여 모델 파라미터를 학습시킨다.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# 테스트 데이터셋에 대하여 모델 성능을 체크하면서 모델이 학습되고 있는지 확인한다.

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 학습 과정은 여러 iteration 수행된다. 각각의 epoch에서 모델은 파라미터가 더 나은 예측을 할 수 있게 학습한다.

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Saving Models

# 모델을 저장하기 위한 일반적인 방법은 internal state dictionary를 저장하는 것이다.

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# Loading Models

# 모델을 로드하는 과정은 모델 구조를 다시 생성하고 state dictionary를 로드하는 과정을 포함한다.

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

# 이 모델은 예측을 만드는 데에 사용될 수 있다.

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
