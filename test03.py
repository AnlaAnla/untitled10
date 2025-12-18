def _binary_to_pil(binary_data):
    """
    将二进制数据转换为PIL Image对象

    Args:
        binary_data: bytes类型的图像二进制数据

    Returns:
        PIL.Image对象
    """
    image = Image.open(io.BytesIO(binary_data))
    image.save('test.png')
    return image


async def process_batch_insert_binary(batch_items, yolo_model, vec_model,
                                      collection, sport, year, program, collection_type, zipUrl):
    if not batch_items:
        return 0

    yolo_inputs = [_binary_to_pil(item['binary']) for item in batch_items]

    try:
        # YOLO 批量预测
        yolo_model.predict_batch(yolo_inputs)
        cropped_images_np = yolo_model.get_max_img_list()

        # 转换为 PIL Image
        pil_images = [Image.fromarray(img) for img in cropped_images_np]
        img_vecs = vec_model.run(pil_images)
    except Exception as e:
        logger.error(f"批量推理阶段失败: {e}")
        return 0


我现在写了这个方法, batch_items传进来的是文件的二进制数组, 上面代码会有问题吗