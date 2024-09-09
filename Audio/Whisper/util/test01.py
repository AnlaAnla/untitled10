from datasets import load_dataset, DatasetDict

from datasets import Audio


common_voice = DatasetDict()
dataset = load_dataset("audiofolder", data_dir=r"D:\Code\ML\Audio\Data1")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print(dataset)