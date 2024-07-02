import glob
import os

import cv2
import numpy as np


def resize_image_and_labels(image_path, label_path, new_size=(1024, 1024)):
    """
    Resize an image while preserving its aspect ratio, and adjust the polygon bounding box coordinates accordingly.

    Args:
        image_path (str): Path to the input image file.
        label_path (str): Path to the input label file with polygon annotations.
        new_size (tuple): Desired new size of the image (width, height).

    Returns:
        resized_image (numpy.ndarray): Resized image with padded background.
        new_labels (list): List of adjusted polygon bounding box coordinates.
    """
    # Load the image and labels
    image = cv2.imread(image_path)
    with open(label_path, 'r') as f:
        labels = [list(map(float, line.strip().split())) for line in f.readlines()]

    # Get the original image size
    height, width = image.shape[:2]

    # Calculate the scale factor and padding
    scale = min(new_size[1] / height, new_size[0] / width)
    new_height = int(height * scale)
    new_width = int(width * scale)
    pad_height = (new_size[1] - new_height) // 2
    pad_width = (new_size[0] - new_width) // 2

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with padded background
    padded_image = np.full(new_size + (3,), 0, dtype=np.uint8)
    padded_image[pad_height:pad_height + new_height, pad_width:pad_width + new_width] = resized_image

    # Adjust the polygon bounding box coordinates
    new_labels = []
    for label in labels:
        class_id = int(label[0])
        polygon = label[1:]
        num_points = len(polygon) // 2

        # Convert the polygon coordinates to relative values
        relative_polygon = [float(x) / width for x in polygon[::2]] + [float(y) / height for y in polygon[1::2]]

        # Scale the polygon coordinates
        scaled_polygon = [x * new_width for x in relative_polygon[::2]] + [y * new_height for y in
                                                                           relative_polygon[1::2]]

        # Adjust the polygon coordinates for padding
        adjusted_polygon = [(x + pad_width) / new_size[0] for x in scaled_polygon[::2]] + [
            (y + pad_height) / new_size[1] for y in scaled_polygon[1::2]]

        new_label = [class_id] + adjusted_polygon
        new_labels.append(new_label)

    return padded_image, new_labels


# 保存调整后的图像和标签
def save_image_and_labels(image, labels, image_path, label_path):
    # cv2.imwrite(image_path, image)
    with open(label_path, 'w') as f:
        for label in labels:
            class_id = int(label[0])
            polygon = [f"{x:.16f}" for x in label[1:]]
            formatted_label = f"{class_id} " + " ".join(polygon)
            f.write(formatted_label + '\n')


'''
# 数据集格式
-train_source
---images
---labels

输出到
-train_target
---images
---labels
'''


def process_image_labels(source_dir, output_dir, new_size=(1024, 1024)):
    images_dir = os.path.join(source_dir, 'images')
    labels_dir = os.path.join(source_dir, 'labels')

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("数据集源文件缺失")

    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    if not os.path.exists(output_images_dir) or not os.path.exists(output_labels_dir):
        os.makedirs(output_images_dir)
        os.makedirs(output_labels_dir)
        print(f"创建: {output_images_dir} | {output_labels_dir}")

    for i, image_name in enumerate(os.listdir(images_dir)):
        label_name = os.path.splitext(image_name)[0] + '.txt'

        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, label_name)
        output_image_path = os.path.join(output_images_dir, image_name)
        output_label_path = os.path.join(output_labels_dir, label_name)

        resized_image, new_labels = resize_image_and_labels(image_path, label_path)

        save_image_and_labels(resized_image, new_labels, output_image_path, output_label_path)

        print(f"{i} 处理: {image_name}\t| {label_name}")


if __name__ == '__main__':
    process_image_labels(r"C:\Code\ML\Image\yolo_data02\Card_2box08\train_source",
                         r"C:\Code\ML\Image\yolo_data02\Card_2box08\train")
    # image_path = 'path/to/your/image.jpg'
    # label_path = 'path/to/your/labels.txt'
    # resized_image, new_labels = resize_image_and_labels(image_path, label_path)
    #
    # # 保存调整后的图像
    # cv2.imwrite('resized_image.jpg', resized_image)
    #
    # # 保存调整后的标签
    # with open('new_labels.txt', 'w') as f:
    #     for label in new_labels:
    #         f.write(' '.join(map(str, label)) + '\n')
