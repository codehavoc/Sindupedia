import librosa
import numpy as np
import os
import csv

# creating header columns of the csv
header = 'filename'

for i in range(1, 16):
    if i == 8 or i == 13:
        continue
    else:
        header += f' mfcc{i}'
header += ' label'
header = header.split()


##############################################################################################################################################################
csvName = 'mfcc.csv'
file = open(csvName, 'w', newline='')

with file:
    writer = csv.writer(file)
    writer.writerow(header)

# genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()  sample of genres
genres = 'classics dance disco hiphop'.split()

print("Strating feature extraction...")

for g in genres:
    for filename in os.listdir(f'Genres/{g}_wavs'):
        songname = f'Genres/{g}_wavs/{filename}'
        duration = int(librosa.get_duration(filename=songname))
        print("\033[1;37m current file :", filename, f'\033[1;31m {duration} Seconds' if duration < 20 else f'{duration} Seconds')

        if duration >= 17:
            y, sr = librosa.load(songname, mono=True, duration=20)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)

            to_append = f'{filename}' 

            for i in range(0,15):
                if i == 7 or i == 12:
                    continue
                else:
                    to_append += f' {np.mean(mfcc[i])}'

            to_append += f' {g}'

            file = open(csvName, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        else:
            print(f'{g} {filename} is defective')

print("ending feature extraction...")

##############################################################################################################################################################