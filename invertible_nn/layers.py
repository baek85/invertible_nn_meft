import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

from typing import Callable, Tuple, Any
import sys
import random
from fractions import Fraction
from invertible_nn.exact_rep import ExactRep
from dataclasses import dataclass

def count_zeros(tensor):
    return torch.count_nonzero(tensor == 0)

#count the number of nonzero fp64 elements become to zero after converting to fp32
def count_fp64(tensor):
    return torch.count_nonzero((tensor.to(torch.float32) == 0) & (tensor != 0))
    

def dynamic_precision(value):
    if value == 0:
        return 4  # 값이 0이면 기본 자릿수 사용
    precision = max(0, 16 - int(torch.log10(torch.abs(value)).item()))
    return precision

def log_tensor_stats(tensor, tensor_name, debug=True):
    # print(f"{tensor_name:10}: Zero count: {count_zeros(tensor):5}, FP64 to FP32 zero count: {count_fp64(tensor):5}")
    if debug:
        # print(f"{tensor_name:10}: Min: {tensor.min():.8f}, Max: {tensor.max():.8f}, Mean: {tensor.mean():.8f}, Std: {tensor.std():.8f}")
        min_val = tensor.min()
        max_val = tensor.max()
        mean_val = tensor.mean()
        std_val = tensor.std()

        # 각 값에 대한 자릿수 동적 설정
        min_precision = dynamic_precision(min_val)
        max_precision = dynamic_precision(max_val)
        mean_precision = dynamic_precision(mean_val)
        std_precision = dynamic_precision(std_val)
        print(f"{tensor_name:10}: Min: {min_val:.{min_precision}f}, "
        f"Max: {max_val:.{max_precision}f}, "
        f"Mean: {mean_val:.{mean_precision}f}, "
        f"Std: {std_val:.{std_precision}f}")
        
def count_unique(tensor, tensor_name):
    unique_values, counts = torch.unique(tensor, return_counts=True)
    print(f"{tensor_name:10}: Unique count: {len(unique_values)}, Unique values: {unique_values}, Counts: {counts}")

RADIX_SCALE = 2**52

def float_to_fixed(x):
    """Converts float to fixed-point integer representation."""
    return (x * RADIX_SCALE).to(dtype=torch.int64)

def detect_fraction_bit_loss(original_value, computed_value):
    original_fixed = float_to_fixed(original_value)
    computed_fixed = float_to_fixed(computed_value)
    if (original_fixed != computed_fixed).any():
        # print 3-dimensional different indices
        diff_indices = torch.nonzero(original_fixed != computed_fixed, as_tuple=False)
        # for i in range(diff_indices.size(0)):
        #     print(f"Loss detected: {diff_indices[i]}")
        return True, diff_indices
    return False, None


def check_fraction_loss(original_value, computed_value, tolerance=1e-10):
    """Checks if there is any loss of precision in fraction bits."""
    difference = abs(original_value - computed_value)
    if difference.min() > tolerance:
        print(f"Loss detected: {difference}")
        return True
    return False

class InvertibleLayer(Function):
    @staticmethod
    def forward(ctx, x):
        if hasattr(x, "save_for_backward"):
            ctx.save_for_backward = x.save_for_backward
            delattr(x, "save_for_backward")


class InvertibleCouplingLayer(Function):

    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """
    
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, F: Callable, G: Callable, first_block: bool) -> torch.Tensor:
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass pass logic.
        forward pass equations:
        Y_1 = X_1 + F(X_2), F = Attention
        Y_2 = X_2 + G(Y_1), G = MLP
        """
        ctx.F = F
        ctx.G = G

        ctx.first_block = first_block
        if hasattr(x, "release_saved_output"):  ## invertible layer 의 출력인지 확인
            x.release_saved_output()  # ctx.saved_tensors 에 있는 레퍼런스 삭제
            ctx.save_output = x.save_output
        ctx.requires_output = hasattr(x, "release_saved_output")  # 다음 레이어야 output 을 저장해야하는지 확인

        # obtaining X_1 and X_2 from the concatenated input
        X_1, X_2 = torch.chunk(x, 2, dim=-1)
        del x

        Y_1 = X_1 + F(X_2)
        del X_1
        Y_2 = X_2 + G(Y_1)
        del X_2
       
        output = torch.cat([Y_1, Y_2], dim=-1)
        del Y_1, Y_2
        # output.save_for_backward = ctx.save_for_backward  ## 이러면 forward 할때 output이 free 될수 있나?
        ## 일단 저장하고, 뒤에 오는게 InvertibleCouplingLayer 타입이면 직접 해제하는 식으로 처리?
        ## 근데 기본적으로 해제될텐데? explicit 하게 저장하도록 하는게 맞는듯
        ## forward 할때 default 는 해제임.
        ## 다음 레이어가 InvertibleCouplingLayer 타입이 아닌지 체크하고, 아니라면 저장하도록 하는게 맞는듯
        ## invertible layer 가 아니면 save_for_backward 호출하도록 하는 방법..?
        # 입력이 invertible 레이어에서 온 것인지 확인
        ctx.output = output.detach()

        def release_saved_output():
            del ctx.output
            # delattr(ctx, 'output')
        def save_output(x):
            ctx.output = x
        output.release_saved_output = release_saved_output
        output.save_output = save_output

        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        y = ctx.output
        F, G = ctx.F, ctx.G

        # obtaining gradients dX_1 and dX_2 from the concatenated input
        Y_1, Y_2 = torch.chunk(y, 2, dim=-1)
        dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad_(True)

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1 = G(Y_1)
            g_Y_1.backward(dY_2)
        Y_1_grad = Y_1.grad
        Y_1 = Y_1.detach()
        g_Y_1 = g_Y_1.detach()

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dX_1 = dY_1 + Y_1_grad

            # free memory since Y_1.grad is now not needed
            del Y_1_grad

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad_(True)

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2 = F(X_2)
            f_X_2.backward(dX_1)
        X_2_grad = X_2.grad
        X_2 = X_2.detach()
        f_X_2 = f_X_2.detach()

        with torch.no_grad():
            dX_2 = dY_2 + X_2_grad
            del X_2_grad

        dx = torch.cat([dX_1, dX_2], dim=-1)

        ## 여기서 다음 backward pass 에 X_1, X_2 를 전달해줘야함
        if ctx.requires_output:
            X_1 = Y_1 - f_X_2
            del f_X_2
            x = torch.cat([X_1, X_2], dim=-1)
            ctx.save_output(x.detach())
        return dx, None, None, None


class CouplingBlock(nn.Module):
    """
    F 랑 G 는 임의의 모듈
    F랑 G를 coupling 구조에 끼워넣음.
    backward pass 할때는 뒷쪽 블락에서 보내준 activation 값을 이용해 중간값 재계산
    Y_1 = X_1 + F(X_2)
    Y_2 = X_2 + G(Y_1)
    """
    def __init__(self, F: nn.Module, G: nn.Module, invert_when_backward=True):
        super().__init__()
        self.F = F
        self.G = G
        self.invert_when_backward = invert_when_backward
        self.first_block = False

    def forward(self, x):
        if self.invert_when_backward:
            return InvertibleCouplingLayer.apply(x, self.F, self.G, self.first_block)
        else:
            X_1, X_2 = torch.chunk(x, 2, dim=-1)
            Y_1 = X_1 + self.F(X_2)
            Y_2 = X_2 + self.G(Y_1)
            return torch.cat([Y_1, Y_2], dim=-1)

class NewInvertibleCouplingLayer(Function):

    """
    Custom Backpropagation function to allow (A) flusing memory in forward
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """
    def __init__(self, fix_random_seed=False):
        self.fix_random_seed = fix_random_seed
        self.random_seeds = {}

    def _init_seed(self, namespace):
        if self.fix_random_seed:
            self.random_seeds[namespace] = random.randint(0, sys.maxsize)
            self._set_seed(namespace)

    def _set_seed(self, namespace):
        if self.fix_random_seed:
            torch.manual_seed(self.random_seeds[namespace])

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, F: Callable, G: Callable, x1_factor: float, x2_factor: float, switch: bool, fix_random_seed=False) -> torch.Tensor:
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass logic.
        forward pass equations:
        y1 = x1_factor * x1 + F(x2), F = Attention
        y2 = x2_factor * x2 + G(y1), G = MLP
        """
        ctx.F = F
        ctx.G = G
        ctx.x1_factor = x1_factor
        ctx.x2_factor = x2_factor
        ctx.switch = switch
        ctx.fix_random_seed = fix_random_seed
        ctx.random_seeds = {}
        ctx.debug = False

        def _init_seed(namespace):
            if fix_random_seed:
                ctx.random_seeds[namespace] = random.randint(0, sys.maxsize)
                _set_seed(namespace)

        def _set_seed(namespace):
            if fix_random_seed:
                torch.manual_seed(ctx.random_seeds[namespace])

        if hasattr(x, "release_saved_output"):  ## invertible layer 의 출력인지 확인
            x.release_saved_output()  # ctx.saved_tensors 에 있는 레퍼런스 삭제
            ctx.save_output = x.save_output
        ctx.requires_output = hasattr(x, "release_saved_output")  # 다음 레이어야 output 을 저장해야하는지 확인

        x1, x2 = torch.chunk(x, 2, dim=-1)
        del x

        _init_seed("F")
        y1 = x1_factor * x1 + F(x2.to(torch.float32)).to(x1.dtype)
        if ctx.debug:
            first_term = x1_factor * x1
            second_term = F(x2.to(torch.float32)).to(x1.dtype)
            result = first_term + second_term
            result = result - second_term
            diff = (result - first_term) / x1_factor
            log_tensor_stats(diff, "x1 diff (forward)")
            ctx.x1 = x1.detach()
            ctx.F_x2 = second_term.detach()
            
        del x1

        _init_seed("G")
        y2 = x2_factor * x2 + G(y1.to(torch.float32)).to(x2.dtype)

        if ctx.debug:
            first_term = x2_factor * x2
            second_term = G(y1.to(torch.float32)).to(x2.dtype)
            result = first_term + second_term
            result = result - second_term
            diff = (result - first_term) / x2_factor
            log_tensor_stats(diff, "x2 diff (forward)")
            ctx.x2 = x2.detach()
            ctx.G_y1 = second_term.detach()
            # breakpoint()

        del x2
        
        if switch:
            output = torch.cat([y2, y1], dim=-1)
        else:
            output = torch.cat([y1, y2], dim=-1)
        del y1, y2
        
        ctx.output = output.detach()

        def release_saved_output():
            del ctx.output

        def save_output(x):
            ctx.output = x
        output.release_saved_output = release_saved_output
        output.save_output = save_output
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        y = ctx.output
        F, G = ctx.F, ctx.G
        x1_factor, x2_factor, switch, fix_random_seed = ctx.x1_factor, ctx.x2_factor, ctx.switch, ctx.fix_random_seed

        def _set_seed(namespace):
            if fix_random_seed:
                torch.manual_seed(ctx.random_seeds[namespace])
                
        if switch:
            y2, y1 = torch.chunk(y, 2, dim=-1)
            dy2, dy1 = torch.chunk(dy, 2, dim=-1)
        else:
            y1, y2 = torch.chunk(y, 2, dim=-1)
            dy1, dy2 = torch.chunk(dy, 2, dim=-1)

        with torch.enable_grad():
            y1.requires_grad_(True)
            _set_seed("G")
            g_y1 = G(y1.to(torch.float32))
            g_y1.backward(dy2)


        y1_grad = y1.grad
        y1 = y1.detach()
        g_y1 = g_y1.detach()

        with torch.no_grad():
            x2 = (y2 - g_y1) / x2_factor
            if ctx.debug:
                log_tensor_stats(x2 - ctx.x2, "x2 diff (backward)")
            del g_y1
            dx1 = dy1 + y1_grad
            del y1_grad

        with torch.enable_grad():
            x2.requires_grad_(True)

            _set_seed("F")
            f_x2 = F(x2.to(torch.float32))
            f_x2.backward(dx1)

        x2_grad = x2.grad
        x2 = x2.detach()
        f_x2 = f_x2.detach()
        
        
        with torch.no_grad():
            dy2 = dy2 * x2_factor
            dx2 = dy2 + x2_grad
            del x2_grad
            dx1 = dx1 * x1_factor

        dx = torch.cat([dx1, dx2], dim=-1)        

        if ctx.requires_output:
            x1 = (y1 - f_x2) / x1_factor
            if ctx.debug:
                log_tensor_stats(x1 - ctx.x1, "x1 diff (backward)")
                breakpoint()
            del f_x2
            x = torch.cat([x1, x2], dim=-1)
            ctx.save_output(x.detach())
        return dx, None, None, None, None, None, None
    

class NewCouplingBlock(nn.Module):
    """
    F and G are arbitrary modules.
    Insert F and G into the coupling structure.
    In the backward pass, use the activation values sent from the subsequent block to recompute intermediate values.
    y1 = x1_factor * x1 + F(x2)
    y2 = x2_factor * x2 + G(y1)
    """
    x1_dtype = torch.float64
    x2_dtype = torch.float64
    block_dtype = torch.float32
    def __init__(self, F: nn.Module, G: nn.Module, invert_when_backward=True, x1_factor: float = 1.0, x2_factor: float = 1.0, switch: bool = False, fix_random_seed=True):
        super().__init__()
        self.F = F.to(self.block_dtype)
        self.G = G.to(self.block_dtype)
        self.invert_when_backward = invert_when_backward
        self.x1_factor = x1_factor
        self.x2_factor = x2_factor
        self.switch = switch
        self.fix_random_seed = fix_random_seed


    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x1 = x1.to(self.x1_dtype)
        x2 = x2.to(self.x2_dtype)

        if self.fix_random_seed:
            torch.manual_seed(random.randint(0, sys.maxsize))

        if self.invert_when_backward:
            y = NewInvertibleCouplingLayer.apply(x, self.F, self.G, self.x1_factor, self.x2_factor, self.switch, self.fix_random_seed)
        else:
            y1 = self.x1_factor * x1 + self.F(x2.to(self.block_dtype))
            y2 = self.x2_factor * x2 + self.G(y1.to(self.block_dtype))
            if self.switch:
                y1 = y1.to(self.x2_dtype)
                y2 = y2.to(self.x1_dtype)
                y = torch.cat([y2, y1], dim=-1)
            else:
                y1 = y1.to(self.x1_dtype)
                y2 = y2.to(self.x2_dtype)
                y = torch.cat([y1, y2], dim=-1)

        return y



class InvertibleResidualLayer(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, F: Callable, max_iter=100, use_anderson_acceleration=False) -> torch.Tensor:
        """
        forward pass equations:
        y = x + F(x)
        F(x) must be 1-Lipschitz
        """
        ctx.F = F
        if hasattr(x, "release_saved_output"):  ## invertible layer 의 출력인지 확인
            x.release_saved_output()  # ctx.saved_tensors 에 있는 레퍼런스 삭제
            ctx.save_output = x.save_output
        ctx.requires_output = hasattr(x, "release_saved_output")  # 다음 레이어야 output 을 저장해야하는지 확인

        output = x + F(x)

        ctx.output = output.detach()
        ctx.max_iter = max_iter
        ctx.use_anderson_acceleration = use_anderson_acceleration

        def release_saved_output():
            del ctx.output
            # delattr(ctx, 'output')
        def save_output(x):
            ctx.output = x
        output.release_saved_output = release_saved_output
        output.save_output = save_output
        return output

    @staticmethod
    def fixed_point_iteration(F, y, max_iter=100, atol=1e-5):
        x = y
        for _ in range(max_iter):
            x = y - F(x)
            if torch.allclose(x, y, atol=atol):
                break
        return x

    @staticmethod
    def anderson_acceleration(F, y, max_iter=100, atol=1e-5, m=5):
        pass

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        y = ctx.output
        F = ctx.F
        max_iter = ctx.max_iter
        use_anderson_acceleration = ctx.use_anderson_acceleration

        # reconstruct x from y
        if use_anderson_acceleration:
            x = InvertibleResidualLayer.anderson_acceleration(F, y, max_iter)
        else:
            x = InvertibleResidualLayer.fixed_point_iteration(F, y, max_iter)

        with torch.enable_grad():
            x.requires_grad_(True)
            F(x).backward(dy)
            dx = x.grad + dy
            x = x.detach()

        if ctx.requires_output:
            ctx.save_output(x.detach())

        return dx, None, None, None


class ResidualBlock(nn.Module):
    def __init__(self, F: nn.Module):
        super().__init__()
        self.F = F

    def forward(self, x):
        return InvertibleResidualLayer.apply(x, self.F, 10, False)


def finite_diff_grad_check():
    # device = torch.device("cuda")
    device = torch.device("cpu")
    input = torch.rand(1, 16, requires_grad=True, dtype=torch.float64, device=device)

    num_blocks = 10
    # mlp = lambda: nn.Sequential(
    #     nn.LayerNorm(8),
    #     nn.Linear(8, 8),
    #     nn.GELU(),
    #     nn.Linear(8, 8)
    # )
    # model = nn.Sequential(*[
    #     CouplingBlock(mlp(), mlp())
    #     for _ in range(num_blocks)
    # ])


    mlp = lambda: nn.Sequential(
        # nn.LayerNorm(8),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
    )
    model = nn.Sequential(*[
        ResidualBlock(mlp())
        for _ in range(num_blocks)
    ])
    def apply_spectral_normalization(module):
        if hasattr(module, "weight"):
            torch.nn.utils.parametrizations.spectral_norm(module, n_power_iterations=50),
    model.apply(apply_spectral_normalization)

    model.to(dtype=torch.float64, device=device)
    
    # @torch.autocast("cuda", dtype=torch.bfloat16)
    # @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def forward_loss_fn(x):
        x = model(x)
        loss = x.norm()
        return loss

    if torch.autograd.gradcheck(forward_loss_fn, input, nondet_tol=1e-5):
        print("Gradient check passed!")


if __name__ == "__main__":
    finite_diff_grad_check()
