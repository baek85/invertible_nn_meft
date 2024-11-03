# invertible-nn
Yet another invertible neural nets.  
This implements building blocks for reversible neural net. Because the layers are reversible, we can avoid caching the intermediate activations to the memory and instead reconstruct them during the backward pass. 

## Installation
```
pip install git+https://github.com/baek85/invertible-nn-meft.git
```

## Examples

### CouplingBlock
`invertible_nn_meft.layers.CouplingBlock` implements the reversible layer using coupling function.  
Coupling function consists of an arbitrary function F and G which performs following transform:  
$$X_1, X_2 \leftarrow \text{split}(X)$$  
$$Y_1 = \lambda_1 \cdot X_1 +  F(X_2)$$
$$Y_2 = \lambda_2 \cdot X_2 + G(Y_1)$$
$$Y \leftarrow [Y_1, Y_2]$$  

Typically, F and G can be a small neural network such as an MLP or a self-attention layer.

### Reversible ViT
A reversible Vision Transformer architecture is implemented by composing the `CouplingBlock` layers. 

```python
from invertible_nn_meft.invertible_vit import InvertibleVisionTransformer

device = torch.device("cuda")

model = InvertibleVisionTransformer(
    depth=12,
    patch_size=(16, 16),
    image_size=(224, 224),
    num_classes=1000
)
model.to(device=device)

input = torch.rand(128, 3, 224, 224, device=device)
output = model(input)
loss = output.norm()
loss.backward()
```

### ResidualBlock

`invertible_nn_meft.layers.ResidualBlock` implements a reversible residual layer.  
This block consists of an arbitrary function F and performs the following transform:  
$$y = x + F(x)$$

F(x) is required to be a 1-Lipschitz function for reversibility. 
We use an MLP and apply spectral normalization to enforce the Lipschitz constraint. 
During backward pass, the input is reconstructed using fixed-point iteration method. 


### Make Pre-trained ViT Reversible with trainable adapter
A MEFT architecture is implemented by composing the `invertible_nn_meft.layers.NewCouplingBlock` layers. 

```python
import torch
from invertible_nn_meft.vision_transformer import vit_base_patch16_224, convert_to_meft


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
```