import librosa
from pydub import AudioSegment
from pydub.utils import which,mediainfo
import os


def audioClipping(song):
    AudioSegment.converter = which('C:/ffmpeg-dupe/bin/ffmpeg.exe')
    type = "prediction"
    savepath = 'features'
    genre = f"{type}."
    count = 0

    timestamps = []
    timestampLength = 20

    print("[SERVER] :: Starting Audio Clipping...")

    duration = int(librosa.get_duration(filename=song))

    print("Duration was ",duration)

    while timestampLength < duration:
        timestamps.append(timestampLength)
        timestampLength += 20

    newAudio = AudioSegment.from_file(song,'webm')

    for x in range(0,len(timestamps)-1):

        t1 = timestamps[x]*1000
        t2 = timestamps[x+1]*1000
        
        clippedWav = newAudio[t1:t2]
        wavcount = f"{genre}{str(count).zfill(4)}.wav"

        # print("saving ",wavcount)

        clippedWav.export(f"{savepath}/{wavcount}", format="wav")
        count = count + 1

    print("[SERVER] :: Ending Audio Clipping...")

    return timestamps


songname ='Gimhane (ගිම්හානේ) @DKM Official x YAKA # Official Audio.mp3'
# path = os.path.join(os.getcwd(),songname)

timestamps = audioClipping(songname)
# print(int(librosa.get_duration(filename=songname)))
# print(mediainfo(songname))

