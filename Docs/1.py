#
# 2022.06.10
#
# Created by 김성찬.
#

#
# Automatic Mixed Precision Examples
#

# 일반적으로 "automatic mixed precision training"은 
# torch.cuda.amp.autocast와 torch.cuda.amp.GradScaler로 학습하는 것을 의미한다.

# torch.cuda.amp.autocast의 인스턴스는 선택된 지역에 대한 autocasting이 가능하게 한다.
# Autocasting은 정확도는 유지하면서 성능을 향상시키기 위해 GPU 연산에 대한 정확도를 자동으로 선택한다.

# torch.cuda.amp.GradScaler의 인스턴스는 gradient scaling의 단계를 편리하게 수행하는 것을 돕는다.
# Gradient scaling은 
# ...

# Typical Mixed Precision Training

model = Net().cuda()
optimzier = optim.SGD(model.parameters(), ...)

# 학습 초반에 GradScaler을 만든다.
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # autocasting을 사용하여 forward pass를 실행한다.
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss. scaled loss에 대해 backward()를 실행하여 scaled gradients를 만든다.
        # autocast를 사용하여 backward pass를 하는 것은 추천하는 방식이 아니다.
        # Backward 연산은 대응하는 forward 연산에서 autocast가 사용한 dtype과 같은 dtype을 사용한다.
        scaler.scale(loss).backward()

        # scaler.step()은 optimizer에 할당된 파라미터의 gradient를 unscale한다.
        # 만약 이러한 gradient가 infs나 NaNs를 포함하지 않는다면, optimizer.step()이 실행된다.
        # 아니면, optimizer.step()은 skip된다.
        scaler.step(optimizer)

        # 다음 iteration을 위해 scale을 업데이트한다.
        scaler.update()

# Working with Unscaled Gradients

# scaler.scale(loss).backward()에 의해 만들어진 모든 gradient는 scale되어 있다.
# 만약 backward()와 scaler.step(optimizer) 사이에서 파라미터의 .grad 속성을 수정하거나 검사하고 싶다면,
# 먼저 이들을 unscale해야 한다.
# 예를 들어, gradient clipping은 gradient의 집합을 그들의 global norm(torch.nn.utils.clip_grad_norm_())이나 
# maximum magnitude(torch.nn.utils.clip_grad_value_())가 사용자가 설정한 threshold 이하가 되도록 조절한다.
# 만약 unscaling 없이 clip을 하려고 하면, gradient의 norm이나 maximum magnitude 또한 scale된다.
# 따라서 사용자가 요구한 threshold가 무의미할 수 있다.

# scaler.unscale_(optimizer)은 optimizer에 할당된 파라미터의 gradient를 unscale한다.
# 만약 모델이 다른 optimizer에 할당된 다른 파라미터를 포함하고 있다면,
# scaler.unscale_(optimizer2)를 따로 실행하여 그러한 파라미터들의 gradient도 unscale해야 한다.

# Gradient clipping

# clipping 전에 scaler.unscale_(optimizer)를 실행하는 것은 unscaled gradients를 clip할 수 있게 한다.

scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # optimizer에 할당된 파라미터의 gradient를 in-place 방식으로 unscale한다.
        scaler.unscale_(optimizer)

        # optimizer에 할당된 파라미터의 gradient가 unscale되었으므로 clip한다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer의 gradient는 이미 unscale했다. 
        # 비록 gradient가 infs나 NaNs를 가진다면 optimizer.step()을 skip하지만
        # scaler.step은 그것들을 unscale하지 않는다.
        scaler.step(optimizer)

        # 다음 iteration을 위해 scale을 update한다.
        scaler.update()

# scaler는 이번 iteration에서 해당 optimizer에 대해 scaler.unscale_(optimizer)가 이미 실행된 것을 기록한다.
# 따라서 scaler.step(optimizer)는 optimizer.step()을 내부적으로 실행하기 전에 
# gradients를 중복해서 unscale하면 안된다는 것을 안다.

# Warning
# unscale_은 optimizer 하나 당 step 실행을 할 때마다 한 번만 실행되어야 한다.
# 그리고 optimizer에 할당된 파라미터에 모든 gradient가 축적된 이후에 실행되어야 한다.
# 각각의 step에서 주어진 optimizer에 대해 unscale_을 두 번 실행하는 것은 RuntimeError을 발생시킨다.

# Working with Scaled Gradients

# 