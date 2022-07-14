from glob import glob
import os
import librosa
from pydub import AudioSegment
from pydub.utils import which,mediainfo


AudioSegment.converter = which('ffmpeg')
type = "snj"
savepath = f'Genres/{type}_wavs'
genre = f"{type}."
count = 295
failedSongs = []

def audioClipping(path,name,fileformat):
    global savepath
    global genre
    global count
    global failedSongs

    song = os.path.join(path,name)

    try:
        timestamps = []
        timestampLength = 20

        print(f"current song : {song}")

        duration = int(librosa.get_duration(filename=song))

        while timestampLength < duration:
            timestamps.append(timestampLength)
            timestampLength += 20

        print(f"timestamps : {timestamps}")

        newAudio = AudioSegment.from_file(song,fileformat)

        for x in range(0,len(timestamps)-1):

            t1 = timestamps[x]*1000
            t2 = timestamps[x+1]*1000
            
            clippedWav = newAudio[t1:t2]
            wavcount = f"{genre}{str(count).zfill(4)}.wav"

            print("saving ",wavcount)

            clippedWav.export(f"{savepath}/{wavcount}", format="wav")
            count = count + 1
    except:
        print("song is defective")
        info = mediainfo(song)
        failedSongs.append({
            'format' : info['format_name'],
            'song' : song
        })


def checkSupportTypes(path,folder,name):
    song = os.path.join(path,name)
    print(name)
    try:
        newAudio = AudioSegment.from_file(song,'webm')
        newAudio.export(os.path.join('downloads_new',folder,name), format="wav")
    except:
        newAudio = AudioSegment.from_file(song,'mp4')
        newAudio.export(os.path.join('downloads_new',folder,name), format="wav")


mainpath = f'downloads/{type}'
downloads = 'downloads'

# for folder in os.listdir(downloads):
#     for songname in os.listdir(os.path.join(downloads,folder)):
#         checkSupportTypes(downloads,folder,songname)

# for songname in os.listdir(mainpath):
#     info = mediainfo(os.path.join(mainpath,songname))
#     print(info['format_name'])
#     print("\n")

for songname in os.listdir(mainpath):
    # available file formats are mp4 and webm
    audioClipping(mainpath,songname,'webm')
    print("\n")

print('---------------------------------------------------------------------------------------------------------')
for i in failedSongs:
    print(i['song'])
    print(i['format'])
    print('\n')