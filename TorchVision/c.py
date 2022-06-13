#
# 2022.06.12.
#
# Created by 김성찬.
#

#
# Datasets
#

# Torchvision은 custom datasets를 만들기 위한 utility class뿐만 아니라,
# torchvision.datasets 모듈 내 built-in datasets를 제공한다.

# Built-in datasets

# 모든 데이터셋은 torch.utils.data.Dataset의 하위 클래스이다.
# 그것들은 __getitem__과 __len__ 메서드가 구현되어 있다.
# 따라서, torch.multiprocessing workers를 사용하여 여러 샘플들을 병렬적으로 로드할 수 있는
# torch.utils.data.DataLoader로 그것들을 전달할 수 있다.

import torch
import torchvision

class Thread:
    def __init__(self):
        self.nThread = 1
    
args = Thread()

imagenet_data = torchvision.datasets.ImageNet("path/to/image_net_root")
data_loader = torch.utils.data.DataLoader(
    imagenet_data,
    batch_size=4,
    shuffle=True,
    num_workers=args.nThreads
)

# Base classes for custom datasets

#