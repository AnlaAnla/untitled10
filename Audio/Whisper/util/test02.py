import torchaudio

from datasets import load_dataset, DatasetDict

common_voice = load_dataset("audiofolder", data_dir=r"D:\Code\ML\Audio\Data1", split="train").train_test_split(test_size=0.2)
common_voice['train'] = load_dataset("audiofolder", data_dir=r"D:\Code\ML\Audio\Data1", split="train")
print(common_voice)
