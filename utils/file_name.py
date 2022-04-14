import glob
files = glob.glob(r'RecognizingPhysicalViolence\dataset\**\*.mp4',recursive=True)
files.extend(glob.glob(r'RecognizingPhysicalViolence\dataset\**\*.avi',recursive=True))
print(len(files))
with open('video.txt','w') as f:
    for file in files:
        f.write(file+ '  1  1 '+'\n')

