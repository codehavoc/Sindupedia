import youtube_dl
import os

urls = {
        'pop' : [],
        'soul' : [],
        'rnb' : [],
        'unknown' : [],
        'snj' : []
    }

def download_song(folder,video_url):
    print(f'\nVideo URL is {video_url}\n')

    video_info = youtube_dl.YoutubeDL().extract_info(
        url = video_url,download=False
    )

    filename = f"{video_info['title']}.mp3"

    options={
        'format' : 'bestaudio/best',
        'keepvideo' : False,
        'outtmpl': f"downloads/{folder}/{filename}",
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])

    print("\nDownload complete... {}\n".format(filename))
    print("-------------------------------------------------------------------------------------------------------------------")


for k,v in urls.items():

    if not os.path.isdir(f'downloads/{k}'):
        os.makedirs(f'downloads/{k}')

    for song in v:
        download_song(k,song)