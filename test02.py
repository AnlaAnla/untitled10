import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D


class People:
    # type{10:素食, 11:肉食}
    def __init__(self):
        self.hp = 10
        self.type = 1


class World:
    # pixel_type{0:空地, 1:草地}
    def __init__(self):
        self.map = np.zeros((100, 100))


def simple_plot():
    """
    simple plot
    """
    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)

    # 打开交互模式
    plt.ion()

    # 循环
    for index in range(50):
        # 清除原有图像
        plt.cla()

        # 设定标题等
        plt.title("title")
        # 网格线
        plt.grid(True)


        data = np.random.randint(0, 255, size=(100, 100, 3))
        plt.imshow(data)

        # 暂停
        plt.pause(0.1)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return


simple_plot()


