import librosa
import numpy as np
import os
import pandas as pd

genres = 'classics dance disco hiphop pop rnb soul snj'.split()
imw = 120
imh = 270
filepath = 'Genres'
csvName = 'genre_mel_specs.csv'
chunk = 100
filecount = 0
firstsave = True
labels = []
mel_specs = []

def saveToCsv():
    global filecount
    global firstsave
    global labels
    global mel_specs
    global csvName

    print(f'starting process of {filecount} files...')
    labelCount = len(labels)
    print(labelCount)
    mel_specs = np.array(mel_specs)
    labels = np.array(labels).reshape(labelCount,1)
    
    df = pd.DataFrame(np.hstack((mel_specs,labels)))

    print("saving current stack of features...")

    if firstsave:
        df.to_csv(csvName, index=False)
    else:
        df.to_csv(csvName, mode='a', index=False, header=False)

    firstsave = False
    filecount = 0
    labels = []
    mel_specs = []

    print("features saved...")




print("starting feature extraction...")

for g in genres: 
    if not firstsave and filecount > 0:
        saveToCsv()
    for file in os.listdir(f'{filepath}/{g}_wavs'):
        
        duration = int(librosa.get_duration(filename=f'{filepath}/{g}_wavs/{file}'))
        print(f'Current file : {file} {duration} Seconds')
        
        if duration >= 17:
            # Loading in the audio file
            y, sr = librosa.load(f'{filepath}/{g}_wavs/{file}',duration=20, sr=22050)

            # Computing the mel spectrograms
            spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
            spect = librosa.power_to_db(spect, ref=np.max)
            

            spect.resize(imh,imw, refcheck=False)
                
            labels.append(g)
            spect = spect.flatten()
            mel_specs.append(spect)

            filecount += 1

            if filecount == chunk:
                saveToCsv()

#check if left over files are still available 
if filecount > 0:
    saveToCsv()

print("ending feature extraction...")

# print("Starting cleaning...")

# mel_specs = pd.read_csv('genre_mel_specs.csv')
# mel_specs = mel_specs.rename(columns={'84480': 'labels'})
# mel_specs.to_csv('genre_mel_specs_new.csv', index=False)

# print("ending cleaning...")

