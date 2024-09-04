import os
import glob
import cv2

Data_dir_path = r"C:\Code\ML\Image\_SEGMENT\1986 Fleer Michael Jordan 57_psa_item_lots"
Save_dir_path = r"C:\Code\ML\Image\_SEGMENT\4corner_1986 Fleer Michael Jordan 57_psa_item_lots"

for score_dir_name in os.listdir(Data_dir_path):
    score_dir_path = os.path.join(Data_dir_path, score_dir_name)
    save_score_dir_path = os.path.join(Save_dir_path, score_dir_name)
    if not os.path.exists(save_score_dir_path):
        os.makedirs(save_score_dir_path)

    score = "{:03d}".format(int(float(score_dir_name) * 10))
    img_num = 0
    for image_name in os.listdir(score_dir_path):
        img_num += 1
        image_path = os.path.join(score_dir_path, image_name)

        print('读取: ', image_path)
        image = cv2.imread(image_path)
        if image.shape[0] == 320 and image.shape[1] == 320:
            break


        height, width, _ = image.shape

        # 截取图片
        img1 = image[height // 4: height // 4 + 320, width // 20: width // 20 + 320:, :]
        img2 = image[height // 4: height // 4 + 320, width - width // 20 - 320: width - width // 20:, :]
        img3 = image[height - height//20-320:height - height//20, width // 20: width // 20 + 320:, :]
        img4 = image[height - height//20-320:height - height//20:, width - width // 20 - 320: width - width // 20:, :]

        img_num_str = "{:03d}".format(img_num)
        img1_name = f"s_{score}_{img_num_str}_1.jpg"
        img2_name = f"s_{score}_{img_num_str}_2.jpg"
        img3_name = f"s_{score}_{img_num_str}_3.jpg"
        img4_name = f"s_{score}_{img_num_str}_4.jpg"

        img1_path = os.path.join(save_score_dir_path, img1_name)
        img2_path = os.path.join(save_score_dir_path, img2_name)
        img3_path = os.path.join(save_score_dir_path, img3_name)
        img4_path = os.path.join(save_score_dir_path, img4_name)

        cv2.imwrite(img1_path, img1)
        cv2.imwrite(img2_path, img2)
        cv2.imwrite(img3_path, img3)
        cv2.imwrite(img4_path, img4)
