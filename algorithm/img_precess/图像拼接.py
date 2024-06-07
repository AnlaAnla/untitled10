from stitching import Stitcher
import cv2
import matplotlib.pyplot as plt

# stitcher = Stitcher()
stitcher = Stitcher(detector="sift", confidence_threshold=0.2)

img1 = cv2.imread(r"C:\Code\ML\Image\Card_test\test\rect1.jpg")
img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

img3 = cv2.imread(r"C:\Code\ML\Image\Card_test\test\rect3.jpg")
img3 = cv2.rotate(img3, cv2.ROTATE_90_COUNTERCLOCKWISE)

panorama1 = stitcher.stitch([img1, img3])
panorama1 = cv2.resize(panorama1, (638, 192))
panorama1 = cv2.rotate(panorama1, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('panorama1', panorama1)

# ==========
img2 = cv2.imread(r"C:\Code\ML\Image\Card_test\test\rect2.jpg")
img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

img4 = cv2.imread(r"C:\Code\ML\Image\Card_test\test\rect4.jpg")
img4 = cv2.rotate(img4, cv2.ROTATE_90_COUNTERCLOCKWISE)

panorama2 = stitcher.stitch([img2, img4])
panorama2 = cv2.rotate(panorama2, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('panorama2', panorama2)

result = stitcher.stitch([panorama1, panorama2])

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
