import math
from matplotlib import pyplot as plt

weight_sigma = 20
x = -10

x_l = []
y_l = []
for i in range(200):
    x += 0.1
    y = math.exp(- (x ** 2) / (2 * weight_sigma ** 2))
    x_l.append(x)
    y_l.append(y)


plt.plot(x_l, y_l)
plt.show()
print(x)


