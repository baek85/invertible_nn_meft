import torch
import torch.backends
from invertible_nn.invertible_vit import InvertibleVisionTransformer
import numpy as np
import random
import os

device = torch.device("cuda")
input = torch.rand(64, 3, 224, 224, requires_grad=True, device=device)
random_seed = 42

# set reproducibility

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed) 


model = InvertibleVisionTransformer(
    depth=24,
    patch_size=(16, 16),
    image_size=(224, 224),
    num_classes=1000,
    mode='vanilla',
    scale_factor=0.1
)
model.to(device=device)

import numpy as np
print(sum([np.prod(p.size()) for p in model.parameters()]))

# @torch.autocast("cuda", dtype=torch.bfloat16)

model.layers[0].first_block = True
# breakpoint()

# for idx, block in enumerate(model.layers):
#     if idx < 6:
#     # if idx < 8:
#         block.invert_when_backward = True
#     else:
#         block.invert_when_backward = False

# model.layers[3].invert_when_backward = False
# model.layers[7].invert_when_backward = False

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

# breakpoint()
# input = torch.rand(1, 3, 224, 224, requires_grad=True, dtype=torch.float64, device=device)
# model.to(dtype=torch.float64, device=device)

# def forward_loss_fn(x):
#     x = model(x)
#     loss = x.norm()
#     return loss

# if torch.autograd.gradcheck(forward_loss_fn, input, nondet_tol=1e-5):
#     print("Gradient check passed!")

