import numpy as np
import matplotlib.pyplot as plt

# 定义元素类型
EMPTY = 0
GRASS = 1
HERBIVORE = 2
CARNIVORE = 3


class Animal:
    def __init__(self, hp, animal_type):
        self.hp = hp  # 动物的能量
        self.type = animal_type  # 动物类型：HERBIVORE 或 CARNIVORE


class World:
    def __init__(self, size=100, grass_growth_rate=0.005):
        self.size = size
        self.grass_growth_rate = grass_growth_rate
        self.herbivore_breed_threshold = 100  # 素食者繁殖临界值
        self.carnivore_breed_threshold = 100  # 肉食者繁殖临界值

        self.map = np.zeros((size, size), dtype=int)  # 初始化地图
        self.animals = {}  # 存储动物对象，键是位置元组，值是 Animal 实例

    def random_spawn(self, count, animal_type):
        """随机生成动物"""
        for _ in range(count):
            while True:
                x, y = np.random.randint(0, self.size, size=2)
                if self.map[x, y] == EMPTY or self.map[x, y] == GRASS:  # 只在空地和草地生成
                    self.map[x, y] = animal_type
                    self.animals[(x, y)] = Animal(hp=10, animal_type=animal_type)
                    break

    def grow_grass(self):
        """随机生成草地"""
        for _ in range(int(self.size * self.size * self.grass_growth_rate)):
            x, y = np.random.randint(0, self.size, size=2)
            if self.map[x, y] == EMPTY:
                self.map[x, y] = GRASS

    def get_neighbors(self, x, y):
        """获取周围 25 个格子的坐标和类型"""
        neighbors = []
        for dx in [-3, -2, -1, 0, 1, 2, 3]:
            for dy in [-3, -2, -1, 0, 1, 2, 3]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.size, (y + dy) % self.size
                neighbors.append((nx, ny, self.map[nx, ny]))
        return neighbors

    def move_animals(self):
        """移动动物并更新状态"""
        new_animals = {}
        for (x, y), animal in list(self.animals.items()):
            if animal.hp <= 0:  # 动物死亡
                self.map[x, y] = EMPTY
                continue

            neighbors = self.get_neighbors(x, y)
            target = None

            if animal.type == CARNIVORE:
                # 肉食者优先寻找素食者
                prey_positions = [(nx, ny) for nx, ny, t in neighbors if t == HERBIVORE]
                if prey_positions:
                    # 朝素食者移动
                    target = prey_positions[0]
            elif animal.type == HERBIVORE:
                # 素食者优先寻找草地，逃避肉食者
                grass_positions = [(nx, ny) for nx, ny, t in neighbors if t == GRASS]
                predator_positions = [(nx, ny) for nx, ny, t in neighbors if t == CARNIVORE]
                if predator_positions:
                    # 有肉食者，远离它们
                    px, py = predator_positions[0]
                    dx, dy = x - px, y - py
                    target = ((x + dx) % self.size, (y + dy) % self.size)
                elif grass_positions:
                    # 没有肉食者，但有草地，朝草地移动
                    target = grass_positions[0]

            # 如果没有目标，随机移动
            if not target:
                dx, dy = np.random.choice([-1, 0, 1], size=2)
                target = ((x + dx) % self.size, (y + dy) % self.size)

            nx, ny = target
            if self.map[nx, ny] == EMPTY:  # 空地移动
                self.map[x, y] = EMPTY
                self.map[nx, ny] = animal.type
                new_animals[(nx, ny)] = animal
            elif self.map[nx, ny] == GRASS and animal.type == HERBIVORE:  # 素食者吃草
                if animal.hp // 2 >= self.herbivore_breed_threshold:  # 素食者繁衍
                    self.map[x, y] = HERBIVORE
                    self.map[nx, ny] = HERBIVORE
                    animal.hp //= 2

                    new_animals[(x, y)] = animal
                    new_animals[(nx, ny)] = Animal(hp=animal.hp // 2, animal_type=HERBIVORE)
                else:
                    self.map[x, y] = EMPTY
                    self.map[nx, ny] = HERBIVORE
                    animal.hp += 10
                    new_animals[(nx, ny)] = animal
            elif self.map[nx, ny] == GRASS and animal.type == CARNIVORE:
                self.map[x, y] = GRASS
                self.map[nx, ny] = CARNIVORE
                new_animals[(nx, ny)] = animal
            elif self.map[nx, ny] == HERBIVORE and animal.type == CARNIVORE:  # 肉食者吃素食者
                self.map[x, y] = EMPTY
                self.map[nx, ny] = CARNIVORE

                if (nx, ny) in self.animals:  # 确保被吃掉的动物在字典中
                    animal.hp += self.animals[(nx, ny)].hp
                    del self.animals[(nx, ny)]

                if animal.hp // 2 >= self.carnivore_breed_threshold:  # 肉食者繁殖
                    animal.hp //= 2

                    new_animals[(x, y)] = animal
                    new_animals[(nx, ny)] = Animal(hp=animal.hp // 2, animal_type=CARNIVORE)
                else:
                    new_animals[(nx, ny)] = animal
            else:  # 无法移动
                new_animals[(x, y)] = animal

            animal.hp -= 1  # 每移动一步消耗能量

            self.animals = new_animals

    def update(self):
        """更新世界状态"""
        self.grow_grass()
        self.move_animals()

    def get_map(self):
        """返回当前地图的彩色 3D 数组"""
        color_map = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # 将不同类型映射为对应颜色
        color_map[self.map == EMPTY] = [220, 220, 200]
        color_map[self.map == GRASS] = [0, 255, 0]  # 绿色
        color_map[self.map == HERBIVORE] = [0, 0, 250]  # 蓝色
        color_map[self.map == CARNIVORE] = [250, 0, 0]  # 红色

        return color_map

    def get_num(self):
        num_list = []
        for i in range(4):
            num_list.append(np.sum(self.map == i))
        return num_list


def simulate():
    world = World(size=200, grass_growth_rate=0.005)
    world.random_spawn(150, HERBIVORE)
    world.random_spawn(50, CARNIVORE)

    is_visible = False

    if is_visible:
        plt.figure(figsize=(16, 8))
        plt.ion()  # 开启交互模式

        # 创建两个子图：左侧地图，右侧曲线图
        ax_map = plt.subplot(1, 2, 1)
        ax_curve = plt.subplot(1, 2, 2)

        herbivore_num = []
        carnivore_num = []

        for step in range(118800):
            world.update()

            _, _, herbivore, carnivore = world.get_num()  # 获取统计数量

            print(f"herbivore: {herbivore}, carnivore: {carnivore}")
            if herbivore == 0 or carnivore == 0:
                break
            herbivore_num.append(herbivore)
            carnivore_num.append(carnivore)

            ax_map.clear()
            ax_map.set_title(f"Step: {step}")
            ax_map.imshow(world.get_map(), cmap="viridis", origin="upper")

            # 绘制曲线
            ax_curve.clear()
            ax_curve.set_title("Animal Population Over Time")
            ax_curve.plot(herbivore_num, label=f"Herbivores [{herbivore}]", color="blue")
            ax_curve.plot(carnivore_num, label=f"Carnivores [{carnivore}]", color="red")
            ax_curve.set_xlabel("Time Step")
            ax_curve.set_ylabel("Population")
            ax_curve.legend()

            plt.pause(0.00001)

        plt.ioff()
        plt.show()

    else:
        # ======非可视化模式========

        for i in range(20, 300, 10):
            for j in range(20, 300, 10):
                herbivore_threshold = i
                carnivore_threshold = j
                world.herbivore_breed_threshold = herbivore_threshold
                world.carnivore_breed_threshold = carnivore_threshold

                for step in range(5000):
                    world.update()

                    _, _, herbivore, carnivore = world.get_num()  # 获取统计数量

                    if step % 200 == 0:
                        print(f"参数[{i},{j}], step: {step}, herbivore: {herbivore}, carnivore: {carnivore}")
                    if herbivore == 0 or carnivore == 0:
                        print('+' * 30)
                        print(f"参数[{i},{j}], result [step: {step}, herbivore: {herbivore}, carnivore: {carnivore}]")
                        print('+' * 30)
                        break


if __name__ == '__main__':
    simulate()
