import os




def replace_label_id(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    new_data = []
    for line in data:
        if line[0] == '1':
            new_line = '2' + line[1:]
        # elif line[0] == '1':
        #     new_line = '0' + line[1:]
        else:
            new_line = line

        new_data.append(new_line)

    with open(label_path, 'w', encoding='utf-8') as f:
        f.writelines(new_data)


if __name__ == '__main__':
    label_dir = r"C:\Code\ML\Image\yolo_data02\series\annotations"

    for label_name in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_name)
        replace_label_id(label_path)
        print(label_path)
