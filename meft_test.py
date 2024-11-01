import torch
from invertible_nn.invertible_vit import InvertibleVisionTransformer
from invertible_nn.vision_transformer import vit_base_patch16_224

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

precision=torch.float32

device = torch.device("cuda")
input = torch.rand(64, 3, 224, 224, requires_grad=True, dtype=precision, device=device)

model = vit_base_patch16_224(pretrained=True, num_classes=1000, mode='meft1', x1_factor=0.1, x2_factor=1.0, reduction_ratio=4)
model.convert_to_meft(mode='meft1', x1_factor=0.5, x2_factor=1.0, reduction_ratio=4)
# model.convert_to_meft(mode='meft2', x1_factor=1.0, x2_factor=0.1, reduction_ratio=4)
# model.convert_to_meft(mode='meft3', x1_factor=0.1, x2_factor=0.1, reduction_ratio=4)
model.to(device=device, dtype=precision)

import numpy as np
print(sum([np.prod(p.size()) for p in model.parameters()]))

# for idx, block in enumerate(model.blocks):
#     if idx in [7, 3]:
#         block.invert_when_backward = False


output = model(input)
# loss = output.norm()
loss = softmax_entropy(output).mean()
# breakpoint()
loss.backward()

invertible_grads = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        invertible_grads[name] = param.grad.clone()
        param.grad = None


for block in model.blocks:
    block.invert_when_backward = False

# input = input.to(dtype=torch.float32)
# model.to(dtype=torch.float32)

output = model(input)
# loss = output.norm()
loss = softmax_entropy(output).mean()
loss.backward()


for name, param in model.named_parameters():
    if param.grad is not None:
        # if 'adapter' in name:
        # caculate angular difference
        cosine = torch.dot(param.grad.flatten(), invertible_grads[name].flatten()) / (
            torch.norm(param.grad) * torch.norm(invertible_grads[name])
        )
        angle = torch.acos(cosine) * 180 / 3.1415926
        if angle > 5:
            print(f"Gradient check for {name} with angle: {angle}")
        # if torch.allclose(param.grad, invertible_grads[name], rtol=1e-5, atol=1e-5):
        #     pass
        #     # print(f"Gradient check passed for {name}!")
        # else:
        #     print(f"Gradient check failed for {name}!")
        #     print(f"Diff: {torch.norm(param.grad - invertible_grads[name])}")
            # print(f"Calculated gradient: {param.grad}")
            # print(f"Saved gradient: {invertible_grads[name]}")

