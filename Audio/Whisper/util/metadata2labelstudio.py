import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    metadata_path = r"D:\Code\ML\Audio\card_audio_data\metadata.csv"
    label_studio_cache_path = r"C:\Users\martin\AppData\Local\label-studio\label-studio\media\upload\22"

    new_metadata = []
    metadata = pd.read_csv(metadata_path)
    metadata = np.array(metadata)

    cache_names = os.listdir(label_studio_cache_path)
    cache_names_split = [name.split('-') for name in cache_names if name.count('-')==1]
    cache_names_split = np.array(cache_names_split)

    for audio_name, text in metadata:
        audio_name = os.path.split(audio_name)[-1]


        cache_audio_index = np.argwhere(cache_names_split == audio_name)

        new_audio_name = "-".join(cache_names_split[cache_audio_index[0][0]])
        new_audio_name = f"/data/upload/{os.path.split(label_studio_cache_path)[-1]}/{new_audio_name}"

        new_metadata.append([new_audio_name, text])
        print(audio_name, ' --> ',new_audio_name, text)

    new_metadata = pd.DataFrame(new_metadata, columns=['audio', 'sentence'])

    data_save_path = os.path.join(os.path.split(metadata_path)[0], 'metadata2labelstudio.csv')
    new_metadata.to_csv(data_save_path, encoding='utf-8', index=False)

    print("处理后保存为:", data_save_path)
