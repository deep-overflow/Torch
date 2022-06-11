#
# 2022.06.11
#
# Created by 김성찬.
#

#
# Save and Load the Model
#

import torch
import torchvision.models as models

# Saving and Loading Model Weights

# 파이토치 모델은 state_dict라고 불리는 학습된 파라미터를 내부 state dictionary에 저장한다.
# torch.save 메서드를 통해 이를 저장할 수 있다.

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), "model_weights.pth")

# 모델 가중치를 로드하기 위해, 먼저 같은 모델의 인스턴스를 만들어야 한다.
# 그리고 load_state_dict() 메서드를 통해 파라미터를 로드한다.

model = models.vgg16()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# inference하기 전에 dropout과 batch normalization layer을 evaluaiton 모드로 설정하기 위해
# model.eval() 메서드를 실행해야 한다.
# 이를 실행하지 않으면 일관되지 않은 inference 결과가 발생한다.

# Saving and Loading Models with Shapes

# 모델 가중치를 로드할 때, 먼저 모델 클래스를 인스턴스화해야 한다.
# 이는 클래스가 네트워크의 구조를 정의하고 있기 때문이다.
# 모델과 함께 이 클래스의 구조를 저장하고 싶은 경우, model.state_dict() 대신 model을 전달할 수 있다.

torch.save(model, "model.pth")

# 그러면 모델을 아래와 같이 로드할 수 있다.

model = torch.load("model.pth")

# 이 방식은 모델을 serialize할 때 파이썬의 pickle 모듈을 사용한다.
# 따라서 모델을 로드할 때 가능한 실제 클래스 정의에 의존해야 한다.