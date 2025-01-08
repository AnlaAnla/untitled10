import random

def generate_terrain_cellular(width, height, iterations=5, tree_prob=0.45):
    """
    使用元胞自动机生成地形

    Args:
        width: 地图宽度
        height: 地图高度
        iterations: 迭代次数
        tree_prob: 初始树木生成概率

    Returns:
        二维列表表示的地形，0 表示地面，1 表示树木
    """

    # 初始化
    terrain = [[1 if random.random() < tree_prob else 0 for _ in range(width)] for _ in range(height)]

    # 定义邻居偏移量
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # 迭代更新
    for _ in range(iterations):
        new_terrain = [row[:] for row in terrain]  # 复制一份 terrain
        for y in range(height):
            for x in range(width):
                tree_count = 0
                for dy, dx in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and terrain[ny][nx] == 1:
                        tree_count += 1

                # 根据规则更新单元格状态
                if terrain[y][x] == 1 and tree_count < 2:
                    new_terrain[y][x] = 0  # 树木太少，变成地面
                elif terrain[y][x] == 0 and tree_count > 3:
                    new_terrain[y][x] = 1  # 地面周围树木足够多，变成树木
                # 可以根据需要添加更多规则

        terrain = new_terrain

    return terrain

# 使用示例
width = 100
height = 50
terrain = generate_terrain_cellular(width, height)

# 打印地形 (为了方便查看，这里只打印一部分)
for y in range(20):
    print("".join(map(str, terrain[y][:20])))