import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()  # 定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

# 定义三维数据
xx = np.arange(-50, 50, 0.05)
yy = np.arange(-50, 50, 0.05)
X, Y = np.meshgrid(xx, yy)
Z = X * Y

# 作图
ax3.plot_surface(X, Y, Z, cmap='rainbow')
# ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
plt.show()
