import torch
from torch import nn
import invertible_nn
import math


class Adapter(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 dropout=0.0,
                 adapter_scalar="learnable_scalar"):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        #_before
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.in_features, self.hidden_features)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.hidden_features, self.out_features)

        self.dropout = dropout
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            # nn.init.normal_(self.down_proj.weight, std=0.02)
            # nn.init.normal_(self.up_proj.weight, std=0.02)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        return up

class MLP_with_Adapter(nn.Module):
    def __init__(self, num_features, drop_path, norm2, mlp, r=4, adapter_scalar="1.0"):
        super().__init__()
        self.drop_path = drop_path
        self.norm2 = norm2
        self.mlp = mlp
        self.adapter_mlp = Adapter(in_features=num_features, hidden_features= num_features // r, out_features=num_features, dropout=0.0, adapter_scalar=adapter_scalar)

    def forward(self, x):
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x)) + self.drop_path(self.adapter_mlp(x))
        return x
    
class Attention_with_Adapter(nn.Module):
    def __init__(self, num_features, drop_path, norm1, attn, r, adapter_scalar="1.0"):
        super().__init__()
        self.drop_path = drop_path
        self.norm1 = norm1
        self.attn = attn
        self.adapter_attn = Adapter(in_features=num_features, hidden_features= num_features // r, out_features=num_features, dropout=0.0, adapter_scalar=adapter_scalar)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.drop_path(self.attn(x)) + self.drop_path(self.adapter_attn(x))
        return x
    
class Block_with_Adapter(nn.Module):

    def __init__(self, num_features, norm1, attn, drop_path, norm2, mlp, r, adapter_scalar="1.0"):
        super().__init__()
        self.norm1 = norm1
        self.attn = attn
        self.drop_path = drop_path
        self.norm2 = norm2
        self.mlp = mlp
        
        self.adapter_attn = Adapter(in_features=num_features, hidden_features= num_features // r, out_features=num_features, dropout=0.0, adapter_scalar=adapter_scalar)
        self.adapter_mlp = Adapter(in_features=num_features, hidden_features= num_features // r, out_features=num_features, dropout=0.0, adapter_scalar=adapter_scalar)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.drop_path(self.attn(x)) + self.drop_path(self.adapter_attn(x))
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x)) + self.drop_path(self.adapter_mlp(x))
        # x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
        # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, r=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.adapter = Adapter(n_embed=dim, down_size=dim // r, dropout=0.0, adapter_scalar="1.0", adapter_layernorm_option="in")

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        residual = residual + x
        x = self.norm2(residual)
        out1 = self.mlp(x)
        out2 = self.adapter(x)
        return residual + out1 + out2
    
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, r=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.adapter = Adapter(n_embed=dim, down_size=dim // r, dropout=0.0, adapter_scalar="1.0", adapter_layernorm_option="in")

    def forward(self, x):
        residual = x
        x = self.norm(x)
        out1, _ = self.attn(x, x, x)
        out2 = self.adapter(x)
        return residual + out1 + out2
    
class MLPBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, r=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.adapter = Adapter(n_embed=dim, down_size=dim//r, dropout=0.0, adapter_scalar="1.0", adapter_layernorm_option="in")

    def forward(self, x):
        residual = x
        x = self.norm(x)
        out1 = self.mlp(x)
        out2 = self.adapter(x)
        return residual + out1 + out2

class MLPSubblock(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """
    def __init__(
        self,
        dim,
        mlp_ratio=4
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        # The reason for implementing autocast inside forward loop instead
        # in the main training logic is the implicit forward pass during
        # memory efficient gradient backpropagation. In backward pass, the
        # activations need to be recomputed, and if the forward has happened
        # with mixed precision, the recomputation must also be so. This cannot
        # be handled with the autocast setup in main training logic.
        return self.mlp(self.norm(x))


class AttentionSubBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """
    def __init__(
        self,
        dim,
        num_heads
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        # using vanilla attention for simplicity. To support adanced attention
        # module see pyslowfast.
        # Note that the complexity of the attention module is not a concern
        # since it is used blackbox as F block in the reversible logic and
        # can be arbitrary.
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        # See MLP fwd pass for explanation.
        x = self.norm(x)
        out, _ = self.attn(x, x, x)
        return out


class InvertibleTransformerBlock(invertible_nn.layers.CouplingBlock):
    def __init__(self, dim, num_heads):
        super().__init__(
            AttentionSubBlock(dim=dim, num_heads=num_heads),
            MLPSubblock(dim=dim)
        )




class MEFT_Block1(invertible_nn.layers.NewCouplingBlock):
    def __init__(self, transformer_block, adapter, x1_factor=0.1, x2_factor=1.0):
        super().__init__(
            F = transformer_block,
            G = adapter,
            X1_factor=x1_factor,
            X2_factor=x2_factor,
            switch=True
        )

class MEFT_Block2(invertible_nn.layers.NewCouplingBlock):
    def __init__(self, transformer_block, adapter, x1_factor=1.0, x2_factor=0.1):
        super().__init__(
            F = adapter,
            G = transformer_block,
            X1_factor=x1_factor,
            X2_factor=x2_factor,
            switch=True
        )

class MEFT_Block3(invertible_nn.layers.NewCouplingBlock):
    def __init__(self, attention, mlp, x1_factor=0.1, x2_factor=0.1):
        super().__init__(
            F = attention,
            G = mlp,
            X1_factor=x1_factor,
            X2_factor=x2_factor,
            switch=False
        )


    
class InvertibleVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        n_head=8,
        depth=8,
        patch_size=(2, 2),  # this patch size is used for CIFAR-10 --> (32 // 2)**2 = 256 sequence length
        image_size=(32, 32),  # CIFAR-10 image size
        num_classes=10,
        mode='vanilla',
        scale_factor=1.0,
        reduction_ratio=8
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
        self.patch_size = patch_size
        self.reduction_ratio = reduction_ratio

        num_patches = (image_size[0] // self.patch_size[0]) * \
            (image_size[1] // self.patch_size[1])
        
        if mode == 'vanilla':
            self.layers = nn.ModuleList(
                [
                    InvertibleTransformerBlock(
                        dim=self.embed_dim,
                        num_heads=self.n_head
                    )
                    for _ in range(self.depth)
                ]
            )
        elif mode == 'meft1':
            self.layers = nn.ModuleList(
                [
                    MEFT_Block1(
                        TransformerBlock(dim=self.embed_dim, num_heads=self.n_head, mlp_ratio=4, r=self.reduction_ratio),
                        Adapter(n_embed=self.embed_dim, down_size=self.embed_dim // self.reduction_ratio, dropout=0.0, adapter_scalar="1.0", adapter_layernorm_option="in"),
                        # AttentionSubBlock(dim=self.embed_dim, num_heads=self.n_head),
                        # MLPSubblock(dim=self.embed_dim, mlp_ratio=4),
                        x1_factor=scale_factor,
                        x2_factor=1.0
                    )
                    for _ in range(self.depth)
                ]
            )
        elif mode == 'meft2':
            self.layers = nn.ModuleList(
                [
                    MEFT_Block2(
                        TransformerBlock(dim=self.embed_dim, num_heads=self.n_head, mlp_ratio=4, r=self.reduction_ratio),
                        Adapter(n_embed=self.embed_dim, down_size=self.embed_dim // self.reduction_ratio, dropout=0.0, adapter_scalar="1.0", adapter_layernorm_option="in"),
                        x1_factor=1.0,
                        x2_factor=scale_factor
                    )
                    for _ in range(self.depth)
                ]
            )
        elif mode == 'meft3':
            self.layers = nn.ModuleList(
                [
                    MEFT_Block3(
                        AttentionBlock(dim=self.embed_dim, num_heads=self.n_head, r=self.reduction_ratio),
                        MLPBlock(dim=self.embed_dim, mlp_ratio=4, r=self.reduction_ratio),
                        x1_factor=scale_factor,
                        x2_factor=scale_factor
                    )
                    for _ in range(self.depth)
                ]
            )

        # Standard Patchification and absolute positional embeddings as in ViT
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim)
        )

        # The two streams are concatenated and passed through a linear
        # layer for final projection. This is the only part of RevViT
        # that uses different parameters/FLOPs than a standard ViT model.
        # Note that fusion can be done in several ways, including
        # more expressive methods like in paper, but we use
        # linear layer + LN for simplicity.
        self.head = nn.Linear(2 * self.embed_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(2 * self.embed_dim)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x += self.pos_embeddings

        # the two streams X_1 and X_2 are initialized identically with x and
        # concatenated along the last dimension to pass into the reversible blocks
        x = torch.cat([x, x], dim=-1)

        for layer in self.layers:
            x = layer(x)

        # aggregate across sequence length
        x = x.mean(1)

        # head pre-norm
        x = self.norm(x)

        # pre-softmax logits
        x = self.head(x)

        # return pre-softmax logits
        return x


def test():
    device = torch.device("cuda")
    input = torch.rand(1000, 3, 224, 224, requires_grad=True, device=device)

    model = InvertibleVisionTransformer(
        depth=64,
        patch_size=(16, 16),
        image_size=(224, 224),
        num_classes=1000
    )
    model.to(device=device)

    import numpy as np
    print(sum([np.prod(p.size()) for p in model.parameters()]))
    
    # @torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        output = model(input)
        loss = output.norm()

    loss.backward()
    breakpoint()


if __name__ == "__main__":
    test()

