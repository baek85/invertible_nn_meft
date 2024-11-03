import torch
from invertible_nn_meft.invertible_vit import InvertibleVisionTransformer

device = torch.device("cuda")
input = torch.rand(64, 3, 224, 224, requires_grad=True, device=device)

model = InvertibleVisionTransformer(
    depth=32,
    patch_size=(16, 16),
    image_size=(224, 224),
    num_classes=1000,
    # mode='vanilla',
    mode='meft3',
    scale_factor=1.0,
    reduction_ratio=8
)
model.to(device=device)

import numpy as np
print(sum([np.prod(p.size()) for p in model.parameters()]))

# for idx, block in enumerate(model.layers):
#     if idx % 5 == 0:
#         block.invert_when_backward = False

# with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
output = model(input)
loss = output.norm()

loss.backward()
# save calculated gradient for each layer
invertible_grads = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        invertible_grads[name] = param.grad.clone()
        param.grad = None

# breakpoint()
for block in model.layers:
    block.invert_when_backward = False

# with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
output = model(input)
loss = output.norm()

loss.backward()

# compare calculated gradient with saved gradient
for name, param in model.named_parameters():
    if param.grad is not None:
        # if 'adapter' in name:
        if torch.allclose(param.grad, invertible_grads[name], rtol=1e-5, atol=1e-5):
            pass
            # print(f"Gradient check passed for {name}!")
        else:
            print(f"Gradient check failed for {name}!")
            print(f"Diff: {torch.norm(param.grad - invertible_grads[name])}")
            # print(f"Calculated gradient: {param.grad}")
            # print(f"Saved gradient: {invertible_grads[name]}")

