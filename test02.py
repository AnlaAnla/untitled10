def fn(x):
    return x - x * 0.05


x = 1e4
for i in range(10):
    x = fn(x)

print(x)
