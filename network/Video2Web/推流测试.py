import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np

app = FastAPI()


class S:
    def __init__(self):
        self.x = 1


@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


def generate_frames():
    while True:
        # img = cv2.imread(r"C:\Users\wow38\Pictures\scenery\background.jpg")
        #
        # # 随机生成模糊核大小和角度
        # kernel_size = np.random.randint(15, 31)
        # angle = np.random.uniform(0, 360)
        # angle_radian = angle * np.pi / 180
        #
        # # 创建线性运动模糊核
        # kernel_v = np.zeros((kernel_size, kernel_size))
        # kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        # kernel_v = kernel_v / kernel_size
        #
        # # 旋转模糊核
        # M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        # kernel_v = cv2.warpAffine(kernel_v, M, (kernel_size, kernel_size))
        #
        # # 应用动感模糊效果
        # frame = cv2.filter2D(img, -1, kernel_v)
        frame = np.random.randint(0, 255, (640, 480))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    s = S()
    print(s.x)
    s.x = 3

    # while True:
    #     data = input('data:')
    #     if data == '1':
    #         uvicorn.run(app, host='0.0.0.0', port=8100)
    #     elif data == 'q':
    #         del app
