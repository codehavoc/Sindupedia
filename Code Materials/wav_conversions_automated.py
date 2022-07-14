import os
from pydub import AudioSegment
import librosa


# change settings according to the dataset ##############################################################################################################
type = "pop"
path1 = f"downloads\{type}"
path2 = f"Genres\{type}_wavs"
genre = f"{type}."               # set to 0 to automate the entier process
#########################################################################################################################################################

songs = os.listdir(path1)
count = 901

print("starting convertion... starting with file",count)

for song in songs:

    print("\nCurrent song :",song)

    timestamps = []
    timestampLength = 20

    duration = int(librosa.get_duration(filename=path1+"/"+song))

    while timestampLength < duration:
        timestamps.append(timestampLength)
        timestampLength += 20

    newAudio = AudioSegment.from_file(path1+"/"+song,'wav')

    for x in range(0,len(timestamps)-1):

        t1 = timestamps[x]*1000
        t2 = timestamps[x+1]*1000
        
        clippedWav = newAudio[t1:t2]
        wavcount = f"{genre}{str(count).zfill(4)}.wav"

        print("saving ",wavcount)

        clippedWav.export(f"{path2}/{wavcount}", format="wav")
        count = count + 1

print("ending convertion...")
