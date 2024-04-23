import cv2

# 读取图像，转灰度图进行检测
img = cv2.imread(r"C:\Users\wow38\Pictures\scenery\alishan-2136879_1920.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sift实例化对象
sift = cv2.SIFT_create()

# 关键点检测
keypoint = sift.detect(img_gray)

# 关键点信息查看
# print(keypoint)  # [<KeyPoint 000001872E1E2960>, <KeyPoint 000001872E1E2B10>]
original_kp_set = {(int(i.pt[0]), int(i.pt[1])) for i in keypoint}  # pt查看关键点坐标
print(original_kp_set)

# 在图像上绘制关键点的检测结果
cv2.drawKeypoints(img, keypoint, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 显示图像
cv2.imshow("img", img)
cv2.waitKey(0)