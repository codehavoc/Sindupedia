import youtube_dl

def validYoutubeLink(link):
    if 'www.youtube.com/watch?' in link:
        return True
    else:
        return False


def run():
    video_url = input("please enter youtube video url:")

    print(validYoutubeLink(video_url))

    video_info = youtube_dl.YoutubeDL().extract_info(
        url = video_url,download=False
    )
    filename = f"{video_info['title']}.mp3"
    options={
        'format':'bestaudio/best',
        'keepvideo':False,
        'outtmpl': f"downloads/{filename}",
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])

    print("Download complete... {}".format(filename))

if __name__=='__main__':
    run()