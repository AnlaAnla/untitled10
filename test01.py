def maxArea(height) -> int:
    maxdata = 0

    height_len = len(height)
    p1 = 0
    p2 = height_len - 1
    while p1 < p2:
        area = (p2 - p1) * min(height[p1], height[p2])
        if area > maxdata:
            maxdata = area

        if height[p1] > height[p2]:
            p2 -= 1
        else:
            p1 += 1

    return maxdata


data = [1, 8, 6, 2, 5, 4, 8, 3, 7]

print(maxArea(data))
