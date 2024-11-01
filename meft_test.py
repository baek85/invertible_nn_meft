import torch
from invertible_nn.invertible_vit import InvertibleVisionTransformer
from invertible_nn.vision_transformer import vit_base_patch16_224

precision=torch.float64

device = torch.device("cuda")
input = torch.rand(64, 3, 224, 224, requires_grad=True, dtype=precision, device=device)

model = vit_base_patch16_224(pretrained=True, num_classes=1000, mode='meft1', x1_factor=0.1, x2_factor=1.0, reduction_ratio=4)
# model.convert_to_meft(mode='meft1', x1_factor=0.1, x2_factor=1.0, reduction_ratio=4)
# model.convert_to_meft(mode='meft2', x1_factor=1.0, x2_factor=0.1, reduction_ratio=4)
model.convert_to_meft(mode='meft3', x1_factor=0.1, x2_factor=0.1, reduction_ratio=4)
model.to(device=device, dtype=precision)

import numpy as np
print(sum([np.prod(p.size()) for p in model.parameters()]))

for idx, block in enumerate(model.blocks):
    if idx in [6]:
        block.invert_when_backward = False

# with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):

output = model(input)
loss = output.norm()
loss = loss
loss.backward()
# save calculated gradient for each layer
invertible_grads = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        invertible_grads[name] = param.grad.clone()
        param.grad = None


for block in model.blocks:
    block.invert_when_backward = False

# input = input.to(dtype=torch.float32)
# model.to(dtype=torch.float32)
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

