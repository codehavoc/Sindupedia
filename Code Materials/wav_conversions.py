import os
from pydub import AudioSegment


# change settings according to the dataset ##############################################################################################################
type = "test"
path1 = f"{type}_songs"
path2 = f"{type}_wavs"
genre = f"{type}."
userselection = 10                  # set to 0 to automate the entier process
#########################################################################################################################################################

songs = os.listdir(path1)
count = len(os.listdir(path2))
options = [[0,20,40,60,80,100],[20,40,60,80,100,120,140],[10,30,50,70,90,110],[10,30,50,70,90,110,130]]

print("starting convertion... starting with file",count)

for song in songs:

    print("\nCurrent song :",song)
    if userselection != 0:
        userselection = int(input("Select timestamps : "))

    if userselection == 0:
        timestamps = options[userselection]
    else:
        timestamps = options[userselection-1] 

    # from wav files
    # newAudio = AudioSegment.from_wav(path1+"/"+song)
    # from mp3 files
    newAudio = AudioSegment.from_mp3(path1+"/"+song)

    for x in range(0,len(timestamps)-1):

        t1 = timestamps[x]*1000
        t2 = timestamps[x+1]*1000
        
        clippedWav = newAudio[t1:t2]
        wavcount = f"{genre}{str(count).zfill(4)}.wav"

        print("saving ",wavcount)

        clippedWav.export(f"{path2}/{wavcount}", format="wav")
        count = count + 1

print("ending convertion...")
