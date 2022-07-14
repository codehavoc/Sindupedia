import librosa
import numpy as np
import os
import csv

filepath = 'Genres'

# creating header columns of the csv
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'

for i in range(1, 21):
    header += f' mfcc{i}'

header += ' label'
header = header.split()


##############################################################################################################################################################
csvName = 'feature_dataset.csv'
file = open(csvName, 'w', newline='')

with file:
    writer = csv.writer(file)
    writer.writerow(header)

genres = 'classics dance disco hiphop pop rnb soul snj'.split()

print("Strating feature extraction...")

for g in genres:
    for filename in os.listdir(f'{filepath}/{g}_wavs'):
        songname = f'{filepath}/{g}_wavs/{filename}'
        duration = int(librosa.get_duration(filename=songname))
        print("\033[1;37m current file :", filename, f'\033[1;31m {duration} Seconds' if duration < 20 else f'{duration} Seconds')

        if duration >= 17:
            y, sr = librosa.load(songname, mono=True, duration=20)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)

            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            
            for e in mfcc:
                to_append += f' {np.mean(e)}'

            to_append += f' {g}'
            
            file = open(csvName, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        else:
            print(f'{g} {filename} is defective')


print("ending feature extraction...")

##############################################################################################################################################################