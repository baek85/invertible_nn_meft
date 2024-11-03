import torch
from invertible_nn.vision_transformer import vit_base_patch16_224, convert_to_meft


precision=torch.float32

device = torch.device("cuda")
input = torch.rand(64, 3, 224, 224, requires_grad=True, dtype=precision, device=device)

model = vit_base_patch16_224(pretrained=True, num_classes=1000)
convert_to_meft(model, mode='meft1', x1_factor=0.5, x2_factor=1.0, reduction_ratio=4)
model.to(device=device, dtype=precision)

for idx, block in enumerate(model.blocks):
    if idx in [7, 3]:
        block.invert_when_backward = False

import numpy as np
print(sum([np.prod(p.size()) for p in model.parameters()]))

output = model(input)
loss = output.norm()
loss.backward()

invertible_grads = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        invertible_grads[name] = param.grad.clone()
        param.grad = None


for block in model.blocks:
    block.invert_when_backward = False


output = model(input)
loss = output.norm()
loss.backward()


for name, param in model.named_parameters():
    if param.grad is not None:

        if torch.allclose(param.grad, invertible_grads[name], rtol=1e-5, atol=1e-5):
            pass
        else:
            print(f"Gradient check failed for {name}!")
            print(f"Diff: {torch.norm(param.grad - invertible_grads[name])}")