#
# 2022.06.11
#
# Created by 김성찬.
#

#
# Automatic Differentiation with torch.autograd
#

# 뉴럴 네트워크를 학습할 때, 가장 자주 사용되는 알고리즘은 back propagation이다.
# 이 알고리즘에서, 파라미터(모델 가중치)는 주어진 파라미터에 대한 loss 함수의 gradient를 따라 갱신된다.

# 이러한 gradient들을 계산하기 위해, 파이토치는 torch.autograd라고 하는 built-in differentiation engine을 가진다.
# 이것은 어떤 계산 그래프에 대한 gradient 자동 연산을 지원한다.

# 가장 간단한 단일 층 뉴럴 네트워크를 고려해보자.

from importlib.metadata import requires
import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Tensors, Functions and Computational graph

# x - * - + - z - CE - loss
#     |   |       |
#     w   b       y

# 이 네트워크에서 w와 b는 최적화해야 하는 파라미터다.
# 따라서 각각의 변수에 대한 loss 함수의 gradient를 계산해야 한다.
# 이를 위해 텐서의 requires_grad 속성을 설정해야 한다.

# requires_grad의 값을 텐서를 만들 때나 x.requires_grad_(True) 메서드를 통해 설정할 수 있다.

# computational graph를 만들려고 하는 함수는 Function 클래스의 객체이다.
# 이 객체는 forward에서 어떻게 함수를 계산하는 지 알고 있다.
# 또한 backward propagation step에서 derivative를 어떻게 계산하는 지 알고 있다.
# backward propagation 함수에 대한 참조는 텐서의 grad_fn 속성에 저장되어 있다.

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Computing Gradients

# 뉴럴 네트워크 내 파라미터 가중치를 최적화하려면, 각각의 파라미터에 대한 loss 함수의 derivative를 계산해야 한다.
# 따라서 고정된 x, y의 값 아래 d_loss/d_w와 d_loss/d_b를 계산해야 한다.
# 이 derivatives를 계산하기 위해 loss.backward()를 실행하면 된다. 이를 통해 w.grad와 b.grad를 얻을 수 있다.

loss.backward()
print(w.grad)
print(b.grad)

# computational graph의 잎 노드에 대해서만 grad 속성을 얻을 수 있다. 
# 이때 해당 노드의 requires_grad 속성이 True로 설정되어야 한다.
# graph 상 다른 노드들에 대하여 gradient가 사용할 수 없을 것이다.

# backward를 사용하는 gradient 연산은 performance의 이유로 주어진 그래프에 대하여 한 번만 실행할 수 있다.
# 만약 같은 그래프에 대하여 backward를 여러 번 실행해야 할 필요가 있다면, backward 실행에서 retain_graph=True를 전달해야 한다.

# Disabling Gradient Tracking

# 기본적으로, requires_grad=True인 모든 텐서들은 그들의 연산 기록을 추적하고 gradient 연산을 지원한다.
# 하지만, 이것이 필요하지 않은 경우가 있다. 예를 들어, 모델을 학습할 때, 그냥 몇몇의 입력 데이터에 대하여 모델을 실행하고 싶을 수 있다.
# 구체적으로, 오직 네트워크에 대하여 forward 연산만 실행하길 원할 수 있다.
# torch.no_grad() 블록을 연산 코드로 감싸는 방식으로 연산을 추적하는 것을 막을 수 있다.

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# 같은 결과를 얻기 위한 다른 방식은 detach() 메서드를 사용하는 것이다.

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

# gradient tracking을 방지하길 원하는 경우가 있다.
# - 뉴럴 네트워크 내 몇몇 파라미터들을 frozen parameters로 표시하기 위해. 이는 finetuning a pretrained network에서 흔한 경우다.
# - forward 연산만을 하는 경우 연산 속도를 높이기 위해. gradient를 추적하지 않는 연산은 더 효율적이다.

# More on Computational Graphs

# 개념적으로, autograd는 데이터의 기록과 Function objects로 구성된 DAG (directed acyclic graph) 의 모든 실행된 연산을 기록한다.
# 이 DAG 내에서, 잎 노드들은 입력 텐서이고, 루트는 출력 텐서이다.
# 루트에서 잎으로 이 그래프를 추적하다보면, chain rule을 통해 자동적으로 gradient를 계산할 수 있다.

# forward 연산에서, autograd는 동시에 두 가지를 실행한다.
# - 결과 텐서를 연산하기 위해 요구되는 연산을 실행한다.
# - DAG에서 연산의 gradient 함수를 유지한다.

# backward 연산은 DAG root에서 .backward()가 실행되면 시작된다. 그러면 autograd는
# - .grad_fn으로부터 gradient를 계산하고
# - 텐서의 .grad 속성에 값을 축적하고
# - chain rule을 이용하여 잎 텐서로 전파한다.

# DAGs are dynamic in PyTorch

# 중요한 것은 그래프가 처음부터 다시 생성되는 것이다; 각 .backward() 실행 이후, autograd는 새 그래프를 채우기 시작한다.
# 이것은 모델에서 control flow 명령문을 사용할 수 있는 이유이다; 모양이나 사이즈 연산 등을 매 iteration마다 바꿀 수 있다.

# Optional Reading: Tensor Gradients and Jacobian Products

# 많은 경우, scalar loss function을 사용하고, 몇몇 파라미터에 대해 gradient를 계산해야 할 필요가 있다.
# 그러나, 출력 함수가 임의의 텐서인 경우가 있다. 이러한 경우, 파이토치는 Jacobian product를 연산할 수 있게 한다.
# 이것은 실제 gradient가 아니다.

# Jacobian matrix를 계산하는 대신, 파이토치는 Jacobian Product를 계산한다.
# 이것은 벡터를 매개변수로 가지는 backward를 통해 실행된다.

inp = torch.eye(5, requires_grad=True)
out = (inp + 1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}\n")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"Second call\n{inp.grad}\n")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"Call after zeroing gradients\n{inp.grad}")

# backward를 같은 변수에 대하여 두 번째 실행하는 경우 gradient의 값이 달라진다.
# 이것은 backward propagation을 실행할 때, 파이토치가 gradient를 축적하기 때문에 발생한다.
# 계산된 gradient의 값이 computational graph의 모든 잎 노드의 grad 속성에 더해진다.
# 적합한 gradient를 계산하고 싶다면, 먼저 grad 속성을 0으로 만들어야 한다.
# 실제 학습에서 optimizer가 이것을 도와준다.

# 이전에 backward() 함수를 파라미터 없이 실행했었다.
# 이것은 본질적으로 backward(torch.tensor(1.0))을 실행하는 것과 동일하다.
# 이는 뉴럴 네트워크 학습 과정에서 loss와 같은 scalr-valued function의 경우 gradient를 계산하는 데 유용한 방식이다.