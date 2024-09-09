import torchaudio

print(torchaudio.__version__)
data = torchaudio.load(r"D:\Code\ML\Audio\Data1\audio\audio1.mp3")
print(data)