import platform
import sys

if __name__ == '__main__':
    if platform.system() == 'Windows':
        import cv2
    else:
        import asdad

    print(cv2.__version__)
