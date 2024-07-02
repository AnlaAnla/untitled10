import cv2
import numpy as np


class Stitcher:

    def __init__(self, ratio=0.75, reprojThresh=4.0, showMatches=False):
        self.ratio = ratio
        self.reprojThresh = reprojThresh
        self.showMatches = showMatches

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 用来返回图片的特征点点集NumPy数组(而不是默认的对象集列表)，及对应的描述特征，128向量。
    def detectAndDescribe(self, image):
        # 将彩色图片转换成灰度图
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        kps, features = descriptor.detectAndCompute(image, None)

        # 通过kp.pt 将结果转换成NumPy数组，里面每个单元是特征点的坐标(x,y)
        kps = np.float32([kp.pt for kp in kps])

        return kps, features

    # 特征匹配，并且得到变换矩阵H
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB):
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()

        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        # 得到原生匹配对象的结果   我们还没进行比较筛选
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        # 筛选过滤，得到的列表matches是A与B匹配的特征点的索引
        matches = []
        for obj1, obj2 in rawMatches:
            # 当最近距离obj1.distance跟次近距离的比值小于ratio值时，保留此匹配对。
            if obj1.distance < obj2.distance * self.ratio:
                # 记录obj1这个匹配结果在featuresA, featuresB中的索引值
                # 即这个匹配是A图哪个位置的点与B图哪个位置的点,把这两个位置记录下来
                matches.append((obj1.trainIdx, obj1.queryIdx))

        # 当筛选后的匹配对数大于4时，计算视角变换矩阵  如果太少不就没必要匹配了吗(说明可能是不相关的图)
        # 最重要的是 变换矩阵H是3*3 矩阵 8个方程 所以至少4对点
        if len(matches) > 4:
            # 从matches中拿到位置，在kps中取出对应的点(x,y)
            # ksp是所有特征点点集 pts是通过匹配后的特征点点集
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵。  cv2.RANSAC 找到最合适的那一对点，来过略掉一些离群点  用来算H矩阵
            # H：3*3矩阵   status：[[1][1][1][1][0][0]] 这样的列表，表示是否匹配成功，或者说是经变换后是否相似/可以接受。
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reprojThresh)

            # 返回 匹配(位置) 变换矩阵 0/1结果(表示是否匹配成功)
            return matches, H, status

        # 匹配对小于4，返回None
        return None

    # 这是一个用来画线 匹配特征点之间的线的函数
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))

                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)  # 画线

        # 返回可视化结果
        return vis

    # 主拼接函数
    def stitch_row(self, Images):
        # 获取输入图片
        imageB, imageA = Images
        # A、B图片的SIFT关键特征点(真正的点集kps)，特征描述子features
        kpsA, featuresA = self.detectAndDescribe(imageA)
        kpsB, featuresB = self.detectAndDescribe(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        matches, H, status = M

        # 将图片A进行视角变换，A_changed是变换后图片。
        # 宽度相加是因为一会要横向拼接，给B留位置   imageA.shape[0]就是A本身的高度，因为要变换的就是A。
        A_changed = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        # 将图片B传入result图片最左端
        A_changed[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检测是否需要显示图片匹配,就是画着匹配横线的图片。
        if self.showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回匹配后的图片 和 画着线的图片
            return A_changed, vis

        # 返回匹配后的图片
        return A_changed

    def stitch_column(self, Images):
        # 获取输入图片
        imageB, imageA = Images
        # A、B图片的SIFT关键特征点(真正的点集kps)，特征描述子features
        kpsA, featuresA = self.detectAndDescribe(imageA)
        kpsB, featuresB = self.detectAndDescribe(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        matches, H, status = M

        # 将图片A进行视角变换，A_changed是变换后图片。
        # 宽度相加是因为一会要横向拼接，给B留位置   imageA.shape[0]就是A本身的高度，因为要变换的就是A。
        A_changed = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0] + imageB.shape[0]))

        # 将图片B传入result图片最左端
        A_changed[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检测是否需要显示图片匹配,就是画着匹配横线的图片。
        if self.showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回匹配后的图片 和 画着线的图片
            return A_changed, vis

        # 返回匹配后的图片
        return A_changed


# 出图顺序：
'''
1. 原始图片A、B；    
2. A经过变化后的图片；  在类代码中 可以注释掉  
3. B与A变化融合后的图片(结果)；  在类代码中 可以注释掉
4. B与A变化融合后的图片(结果)、画线匹配图(showMatches=True的话)
'''

# 读取拼接图片并显示
# imageA = cv2.imread(r"C:\Code\ML\Image\Card_test\test\c2.jpg")
# imageB = cv2.imread(r"C:\Code\ML\Image\Card_test\test\c1.jpg")
img1 = cv2.imread(r"C:\Code\ML\Image\Card_test\test\rect1.jpg")
img2 = cv2.imread(r"C:\Code\ML\Image\Card_test\test\rect2.jpg")
img3 = cv2.imread(r"C:\Code\ML\Image\Card_test\test\rect3.jpg")
img4 = cv2.imread(r"C:\Code\ML\Image\Card_test\test\rect4.jpg")


# 把图片拼接成全景图
stitcher = Stitcher(showMatches=False)
# result, vis = stitcher.stitch_column([imageA, imageB])
result_top = stitcher.stitch_column([img1, img3])
result_bottom = stitcher.stitch_column([img2, img4])

cv2.imshow('rect1', img1)
cv2.imshow('rect2', img2)
cv2.imshow('rect3', img3)
cv2.imshow('rect4', img4)

# 显示图片
cv2.imshow("result_top", result_top)
cv2.imshow("result_bottom", result_bottom)


result = stitcher.stitch_row([result_top, result_bottom])
cv2.imshow("Result", result)
# cv2.imshow("Keypoint Matches", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()