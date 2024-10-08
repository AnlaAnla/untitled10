import os


def set_file_hidden(file_path):
    # 使用 os 库来设置文件的隐藏属性
    os.system(f'attrib +h "{file_path}"')


def build_show(father_dir):
    """
    该函数建立目标目录的结构为：
    --漫画文件夹
        ----漫画一
            ------001.jpg, 002.jpg, 003.jpg, 004.jpg, 005.jpg ......
        ----漫画二
            ------001.jpg, 002.jpg, 003.jpg, 004.jpg, 005.jpg ......
    ...
    ...
        ----漫画n
    """
    # 如果之前存在这个文件，就删除
    if os.path.exists(os.path.join(father_dir, "show.html")):
        os.remove(os.path.join(father_dir, "show.html"))

    for dir_name in os.listdir(father_dir):
        print(dir_name)

    with open("漫画显示的html代码.txt", "r", encoding="utf-8") as f:
        html = f.read()

    data = []
    for i, dir_name in enumerate(os.listdir(father_dir)):
        dir_path = os.path.join(father_dir, dir_name)
        image_names = [name for name in os.listdir(dir_path) if
                       name.endswith('.jpg') or name.endswith('.png') or name.endswith('.jpeg')]
        data.append([i, dir_name, image_names])

    # print(data)

    # 替换 <%replace01%> 是间隔， 02是数据
    html = html.replace("<%replace01%>", str(int(len(os.listdir(father_dir)) / 14 * 1000)))
    html = html.replace("<%replace02%>", str(data))

    with open(os.path.join(father_dir, "show.html"), 'w', encoding="utf-8") as f:
        f.write(html)

    set_file_hidden(os.path.join(father_dir, "show.html"))

    print("end")

# mydir = r"D:\College\college\image\漫画作品"
# for name in os.listdir(mydir):
#     build_show(os.path.join(mydir, name))
build_show(r"D:\College\college\image\单独下载作品")
