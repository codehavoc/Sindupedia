import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# change settings according to the dataset
imh = 120
imw = 270

genres = 'classics dance disco hiphop pop rnb soul snj'.split()

for g in genres:
    path1 = f"Genres/{g}_wavs"
    path2 = f"Genres/{g}_specs"

    songs = os.listdir(path1)
    images = os.listdir(path2)
    chunks1 = [songs[x:x+20] for x in range(0, len(songs), 20)]
    chunks2 = [images[x:x+20] for x in range(0, len(images), 20)]

    if(len(chunks2)>0):
        print(len(chunks2[len(chunks2)-1]))

        if(len(chunks2[len(chunks2)-1])>=20):
            chunks1=chunks1[len(chunks2):]
        else:
            chunks1=chunks1[len(chunks2)-1:]

    print("starting form file", chunks1[0][0])

    for item in chunks1:
        for i in item:
            plt.rcParams["figure.figsize"] = [7.50, 4.60]
            plt.rcParams["figure.autolayout"] = True
            
            y, sr = librosa.load(f"{path1}/{i}",duration=20, sr=22050) # your file
            plt.axis('off')

            S = librosa.feature.melspectrogram(y=y, sr=sr,  n_fft=2048, hop_length=1024)
            S = librosa.power_to_db(S, ref=np.max)

            S.resize(imh,imw, refcheck=False)

            #to check the size of image in array format
            # S = S.flatten()
            # print(len(S))

            librosa.display.specshow(S)

            imagename = re.sub(".wav", ".png", i)
            plt.savefig(f"{path2}/{imagename}")

            print("file saved", imagename)

    print(f'{g} wav convertions complete...')

print("All conversions complete...")