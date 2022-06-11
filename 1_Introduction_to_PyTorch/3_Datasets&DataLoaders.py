#
# 2022.06.10
#
# Created by 김성찬.
#

#
# Datasets & DataLoaders
#

# 데이터 샘플들을 처리하기 위한 코드는 난잡하고 유지하기 어려울 수 있다.
# 더 나은 가독성과 모듈화를 위해 데이터셋 코드를 모델 학습 코드를 분리하길 원한다.
# 파이토치는 두 개의 data primitives를 제공한다: torch.utils.data.DataLoader와 torch.utils.data.Dataset을 통해
# pre-loaded datasets나 custom datasets를 사용할 수 있도록 한다.

# Dataset : 샘플과 샘플에 대응하는 레이블을 저장한다.
# DataLoader : Dataset을 iterable로 감싸서 샘플에 쉽게 접근할 수 있도록 한다.

# 파이토치 도메인 라이브러리는 torch.utils.data.Dataset을 상속하는 다수의 pre-loaded datasets를 제공하고 특정한 데이터에 대해 구체화된 함수를 구현한다.

# Loading a Dataset

from filecmp import cmp
from torchvision.io import read_image
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Iterating and Visualizing the Dataset

# Datasets를 리스트처럼 인덱싱할 수 있다: training_data[index].
# matplotlib를 통해 몇 개의 샘플을 시각화할 수 있다.

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Creating a Custom Dataset for your files

# custom dataset class는 반드시 3개의 함수를 구현해야 한다: __init__, __len__, __getitem__

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# __init__ : __init__ 함수는 데이터셋 객체를 인스턴스화할 때 한 번 실행된다.
# __len__ : __len__ 함수는 데이터셋의 샘플의 수를 반환한다.
# __getitem__ : __getitem__ 함수는 주어진 인덱스 idx에 대한 샘플을 로드하고 반환한다.
# 인덱스에 기반하여 디스크에서의 이미지의 위치를 확인하고, read_image를 통해 텐서로 변환하고,
# 대응하는 레이블을 csv 데이터로부터 찾는다.
# 적용할 수 있는 transform 함수들을 적용하고 텐서 이미지와 대응하는 레이블을 튜플의 형태로 반환한다.

# Preparing your data for training with DataLoaders

# Dataset은 한 번에 하나의 feature와 label을 반환한다.
# 반면에 모델을 학습하는 동안, 우리는 일반적으로 미니배치의 형태로 샘플들을 전달하고,
# 모델이 과대적합되지 않도록 데이터를 다시 섞고,
# 데이터를 빠르게 처리하기 위해 파이썬의 multiprocessing을 사용하기를 원한다.

# DataLoader은 쉬운 API를 통해 이러한 어려움을 해결할 수 있는 iterable이다.

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader

# 데이터셋을 DataLoader에 올리면 데이터셋이 필요할 때 iterate할 수 있다.
# 매 iteration마다 배치 단위의 train_features와 train_labels를 반환한다.
# shuffle=True로 설정했기 때문에 모든 배치를 iterate하면 데이터를 다시 섰는다.

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

