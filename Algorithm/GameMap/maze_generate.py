import random

def generate_maze_recursive_backtracker(width, height):
    """
    使用递归回溯算法生成迷宫

    Args:
        width: 迷宫宽度 (必须是奇数)
        height: 迷宫高度 (必须是奇数)

    Returns:
        二维列表表示的迷宫，0 表示通路，1 表示墙
    """

    # 确保迷宫的宽度和高度是奇数
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    # 初始化迷宫，全部填充为墙 (1)
    maze = [[1 for _ in range(width)] for _ in range(height)]

    # 定义移动方向 (包括斜向)
    directions = [(0, 2), (0, -2), (2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def carve_passage(x, y):
        """
        从 (x, y) 开始雕刻路径
        """
        maze[y][x] = 0  # 将当前单元格标记为通路

        random.shuffle(directions)  # 打乱方向顺序

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # 检查新坐标是否在迷宫范围内且是墙
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                # 凿开当前单元格和新单元格之间的墙
                maze[ny - dy // 2][nx - dx // 2] = 0
                carve_passage(nx, ny)

    # 从起点开始雕刻路径 (这里选择左上角)
    start_x, start_y = 1, 1
    carve_passage(start_x, start_y)

    # 确保起点和终点是通路
    maze[start_y][start_x] = 0
    maze[height - 2][width - 2] = 0

    return maze



if __name__ == '__main__':

    # 生成一个 41x41 的迷宫
    maze_width = 41
    maze_height = 41
    maze = generate_maze_recursive_backtracker(maze_width, maze_height)

    print(maze)
    # 打印迷宫
    # for row in maze:
        # print("".join(map(str, row)))

    # 起点和终点 (你可以修改这些值)
    start = (1, 1)
    end = (maze_width - 2, maze_height - 2)

    print(f"\nStart: {start}")
    print(f"End: {end}")
