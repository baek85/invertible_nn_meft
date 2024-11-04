import torch

# 예시로 아주 작은 수를 사용해 봅시다
x1 = torch.tensor(1e-36).to(torch.float32)
x2 = torch.tensor(1e-18).to(torch.float32)
f_x2 = x2 * x2  # 예를 들어 f(x2) = x2^2라고 가정

# x1 + f(x2) - f(x2) 계산
result = x1 + f_x2 - f_x2

# 결과 출력
print("x1:", x1, "x2:", x2)
print(f"f(x2) = x2^2: {f_x2}")
print("x1 + f(x2):", x1 + f_x2)
print("x1 + f(x2) - f(x2):", x1 + f_x2 - f_x2)
print("x1 + f(x2) - f(x2) == x1:", result == x1)


from fractions import Fraction

x = 0.003  # 임의의 부동 소수점
fraction = Fraction(x).limit_denominator()
n = fraction.numerator
d = fraction.denominator
print(x, "is approximately", n, "/", d)