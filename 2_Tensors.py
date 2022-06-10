#
# 2022.06.10
#
# Created by 김성찬.
#

#
# Tensors
#

# Tensors는 배열, 행렬과 매우 유사한 특별한 자료 구조다.
# 파이토치에서 모델의 입력과 출력 또는 모델의 파라미터를 텐서를 통해 인코딩한다.

# 텐서는 넘파이의 다차원 배열과 유사한데, GPU 또는 다른 하드웨어 가속기에서 실행할 수 있다는 차이점이 있다.
# 사실 텐서와 넘파이 배열은 같은 메모리를 공유하여 데이터를 복사할 필요를 없게 한다.
# 텐서는 자동 미분에 최적화되어 있다.

import torch
import numpy as np

# Initializing a Tensor

# 텐서는 다양한 방식으로 초기화할 수 있다.

# Directly from data : 텐서는 데이터로부터 직접적으로 생성될 수 있다. 데이터 타입은 자동으로 추론된다.

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From a Numpy array : 텐서는 넘파이 배열로부터 생성될 수 있다.

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor : 새로운 텐서는 매개변수 텐서의 속성을 얻는다.

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# With random or constant values: shape은 텐서 차원의 튜플이다. 아래 함수에서 출력 텐서의 차원을 결정한다.

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tenssor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Attributes of a Tensor

# 텐서의 속성은 shape, datatype, 텐서가 저장된 device를 표현한다.

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Operations on Tensors

# 텐서 연산들은 GPU에서 실행할 수 있다. (일반적으로 CPU에서보다 빠르다.)

# 기본적으로, 텐서는 CPU에서 생성된다. '.to' 메서드를 사용하여 텐서를 명시적으로 GPU로 이동시킬 필요가 있다.
# 큰 텐서들을 device 간에 복사하는 것이 시간과 메모리 측면에서 큰 비용이 들 수 있다는 것을 명시해라.

if torch.backends.mps.is_available():
    tensor = tensor.to("mps")

# Standard numpy-like indexing and slicing

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Joining tensor

# torch.cat을 사용하여 주어진 차원을 따라 텐서들을 연결할 수 있다.
# torch.stack은 torch.cat과 조금 다른 텐서 결합 연산이다.

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic operations

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single-element tensors

# one-element tensor가 있다면, 이것을 item()을 통해 파이썬 수치적인 값으로 바꿀 수 있다.

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations

# 결과를 operand에 저장하는 연산을 in-place라고 부른다. 이들은 '_' suffix를 통해 표시한다.
# 예를 들어, x.copy_(y), x.t_()는 x를 바꿀 것이다.

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# In-place 연산은 메모리를 절약할 수 있다. 하지만 derivatives를 계산할 때 중간 기록의 손실 때문에 문제가 발생한다.

# Bridge with NumPy

# CPU의 텐서와 넘파이 배열은 메모리 공간을 공유한다. 따라서 하나를 수정하면 다른 하나도 수정된다.

# Tensor to NumPy array

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# 텐서에서의 변화는 넘파이 배열에 반영된다.

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor

n = np.ones(5)
t = torch.from_numpy(n)

# 넘파이 배열에서의 변화는 텐서에 반영된다.

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

