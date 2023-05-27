import os

# Get the list of files in the folder
folder_path = r"D:\Code\ML\images\Mywork3\card_database\mosaic\20-21"
files = os.listdir(folder_path)

# Iterate through the files and rename them if they contain "#"
for file in files:
    if "#" in file:
        new_file = file.replace("#", "")
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file))

print('end')