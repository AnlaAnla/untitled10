from compreface.collections.face_collections import FaceCollection
from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections.face_collections import Subjects
import os

DOMAIN: str = 'http://100.64.0.23'
PORT: str = '8000'
RECOGNITION_API_KEY: str = '679508ab-cb67-47ad-8f5f-9bfe5700868c'

compre_face: CompreFace = CompreFace(DOMAIN, PORT, {
    "limit": 0,
    "det_prob_threshold": 0.8,
    "prediction_count": 1,
    "face_plugins": "calculator,age,gender,landmarks",
    "status": "true"
})

recognition: RecognitionService = compre_face.init_face_recognition(
    RECOGNITION_API_KEY)

face_collection: FaceCollection = recognition.get_face_collection()
subjects: Subjects = recognition.get_subjects()

def insert_collection(face_collection, dir_path: str, save_img_num) -> None:
    face_ids = set()
    names = os.listdir(dir_path)

    # 选择每个类别的保存数量
    save_img_num = save_img_num

    for filename in names:
        face_ids.add(filename.split('_')[0])



    temp_num = save_img_num
    for filename in names:
        temp_id = filename.split('_')[0]
        if temp_id not in face_ids:
            continue

        if temp_id in face_ids and temp_num > 0:
            image_path = os.path.join(dir_path, filename)
            subject = temp_id

            # 插入数据库信息
            face_collection.add(image_path, subject)
            print(temp_id, ': ', image_path)
            temp_num -= 1
        else:
            face_ids.remove(temp_id)
            temp_num = save_img_num

    print(face_ids)
    print('end')

if __name__ == '__main__':


    # print(face_collection.list())
    # image_path = r"C:\Code\ML\Image\face_data\65_The CNBC Face Da.ta.ba.se\Caucasian\CF0001_1101_00F.jpg"
    # subject = '666'
    dir_path = r"C:\Code\ML\Image\face_data\65_The CNBC Face Da.ta.ba.se\Asian"
    insert_collection(face_collection, dir_path, 4)

    # face_collection.add(image_path, subject)