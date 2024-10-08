# -*- coding: utf-8 -*-
# @Time : 2023/10/12 13:48
# @Author : XyZeng

import os
import gradio as gr


def get_img_lits(img_dir):
    imgs_List = [os.path.join(img_dir, name) for name in sorted(os.listdir(img_dir)) if
                 name.endswith(('.png', '.jpg', '.webp', '.tif', '.jpeg'))]
    return imgs_List


def input_text(dir):
    img_paths_list = get_img_lits(dir)  # 注意传入自定义的web
    # 结果为 list,里面对象可以为
    dict_path = []
    for i in range(len(img_paths_list)):
        dict_path.append((img_paths_list[i], 'img_descrip' + str(i)))  # 图片路径，图片描述, 图片描述可以自定义字符串
    print(dict_path)
    return dict_path


'''
 gr.Gallery()
 必须要使用.style()才能控制图片布局
 https://www.gradio.app/docs/gallery
 As output: expects a list of images in any format, List[numpy.array | PIL.Image | str | pathlib.Path], 
 or a List of (image, str caption) tuples and displays them.

'''
with gr.Blocks() as iface:
    with gr.Row():
        name_input = gr.Textbox(label="输入数据集名称", placeholder="Enter a name for the directory")
        gallery = gr.Gallery(label="最终的结果图片", columns=[4])

    submit_btn = gr.Button("Submit", variant='primary')
    submit_btn.click(input_text,
                     inputs=name_input,
                     outputs=gallery)

iface.launch()

# demo = gr.Interface(
#     fn=input_text,
#     inputs=gr.Textbox(label='./选择目录'),
#     outputs=gr.Gallery(label="最终的结果图片", columns=[4]),
#     title='显示某路径下的所有图片的缩略图23.10.12',
# )
# if __name__ == "__main__":
#     print("gradio_version", gr.__version__)
#
#     demo.launch(
#         # server_name="0.0.0.0",  # 不指定默认是只能本机 127.0.0.1访问，指定后可局域网访问
#         # server_port=7862    #  可指定端口，好处是固定，坏处是可能占用，默认自动纷纷端口
#     )
