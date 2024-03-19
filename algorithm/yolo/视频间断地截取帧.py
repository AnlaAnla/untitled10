import os
import cv2
import threading


def get_frames(video_dir, save_dir):
    threads = []  # 存放线程
    for video_name in os.listdir(video_dir):
        media_path = os.path.join(video_dir, video_name)

        cap = cv2.VideoCapture(media_path)

        num = 0  # 计算帧数
        frame_space = 100  # 截取帧间隔

        t = threading.Thread(target=save_frame, args=(cap, video_name, save_dir, num, frame_space))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def save_frame(cap, video_name, save_dir, num, frame_space):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("not ret")
            break

        # 截取部分帧
        num += 1
        if num % frame_space == 0:
            cv2.imwrite("{}/{}-{}.jpg".format(save_dir, video_name, num), frame)
            print(video_name, num)

        # 显示图片
        # cv2.imshow("video", frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    video_dir = r"C:\Users\wow38\Videos\Captures"
    save_dir = r"C:\Code\ML\Image\yolov8_data\video_frame\temp"

    get_frames(video_dir, save_dir)
