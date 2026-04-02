import sympy as sp

# 定义符号变量
x = sp.symbols('x')

# 定义函数 y = 2x
f = 3* x ** 5

# 计算函数的导数（微分）
f_prime = sp.diff(f, x)

print(f"函数 y = 2x 的导数是: {f_prime}")