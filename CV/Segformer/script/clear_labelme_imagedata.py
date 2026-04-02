from pathlib import Path
import json

def clear_labelme_imagedata(root_dir: str, backup: bool = True) -> None:
    """
    清除目录下所有 Labelme JSON 文件中的 imageData 字段。

    参数:
        root_dir: 标注数据根目录
        backup: 是否先生成 .bak 备份文件
    """
    root = Path(root_dir)
    if not root.exists():
        print(f"目录不存在: {root}")
        return

    json_files = list(root.rglob("*.json"))
    if not json_files:
        print("没有找到任何 json 文件")
        return

    success_count = 0
    skip_count = 0
    fail_count = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 只有存在 imageData 字段时才处理
            if "imageData" not in data:
                skip_count += 1
                print(f"[跳过] {json_file} -> 不含 imageData 字段")
                continue

            # 如果本来就是空，也跳过
            if data["imageData"] is None:
                skip_count += 1
                print(f"[跳过] {json_file} -> imageData 已为空")
                continue

            if backup:
                backup_file = json_file.with_suffix(json_file.suffix + ".bak")
                if not backup_file.exists():
                    with open(backup_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

            data["imageData"] = None

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            success_count += 1
            print(f"[完成] {json_file}")

        except Exception as e:
            fail_count += 1
            print(f"[失败] {json_file} -> {e}")

    print("\n处理完成")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")


if __name__ == "__main__":
    # 改成你的目录
    folder_path = r"C:\Code\ML\Image\_SEGMENT\video_card_hand_project\labels"

    clear_labelme_imagedata(folder_path, backup=True)