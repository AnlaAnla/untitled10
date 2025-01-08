import heapq
import matplotlib.pyplot as plt
import numpy as np


class Node(object):
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # 起点到节点 n 的实际代价
        self.h = 0  # 节点 n 到终点的估计代价
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)


def astar(maze, start, end, heuristic):
    """
        返回从起点到终点的路径

        Args:
            maze: 二维列表，表示迷宫，0表示可通行，1表示障碍物
            start: 元组，表示起点坐标 (x, y)
            end: 元组，表示终点坐标 (x, y)
            heuristic: 启发函数，接受两个参数：当前节点坐标和终点坐标，返回估计代价

        Returns:
            如果找到路径，返回路径列表，每个元素是一个坐标元组；否则返回 None
    """

    start_node = Node(start)
    end_node = Node(end)

    open_list = []
    closed_list = set()

    # 将起点加入 open list
    heapq.heappush(open_list, start_node)

    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    # moves = [(0, -1), (0, 1), (-1, 0), (1, 0),
    #          (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 循环，直到找到终点或 open list 为空
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # 反转路径

        # 遍历当前节点的邻居节点
        for move in moves:
            neighbor_position = (current_node.position[0] + move[0], current_node.position[1] + move[1])

            # 检查邻居节点是否在迷宫范围内，是否可通行
            if (
                    neighbor_position[0] < 0 or
                    neighbor_position[0] > (len(maze) - 1) or
                    neighbor_position[1] < 0 or
                    neighbor_position[1] > (len(maze[0]) - 1) or
                    maze[neighbor_position[0]][neighbor_position[1]] != 0
            ):
                continue

            # 创建邻居节点
            neighbor_node = Node(neighbor_position, current_node)

            # 如果邻居节点已经在 closed list 中，则跳过
            if neighbor_node in closed_list:
                continue

            # 计算邻居节点的 g, h, f 值
            neighbor_node.g = current_node.g + 1
            neighbor_node.h = heuristic(current_node.position, end_node.position)
            neighbor_node.f = neighbor_node.g + current_node.h

            # 如果邻居节点已经在 open list 中
            if neighbor_node in open_list:
                for open_node in open_list:
                    if neighbor_node == open_node and neighbor_node.g < open_node.g:
                        open_node.g = neighbor_node.g
                        open_node.parent = neighbor_node.parent
                        open_node.f = neighbor_node.f
            else:
                heapq.heappush(open_list, neighbor_node)

    # 如果 open list 为空，则表示没有找到路径
    return Node


# 启发函数示例 (曼哈顿距离)
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclid_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def show_maze(maze, path):
    for p in path:
        maze[p[0]][p[1]] = 8
    maze = np.array(maze)
    plt.imshow(maze)
    plt.show()


# 使用示例
from maze_generate import generate_maze_recursive_backtracker

maze = generate_maze_recursive_backtracker(50, 50)
# maze = [
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
#     [0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
#     [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
# ]

start = (0, 0)
end = (40, 40)

path = astar(maze, start, end, manhattan_distance)

if path:
    print(f"找到从 {start} 到 {end} 的路径:")
    for p in path:
        print(p)
else:
    print(f"没有找到从 {start} 到 {end} 的路径")

show_maze(maze, path)
