import requests
import re
import os

# contains a youtube url and a normal website url ]
url = ['https://www.learningcontainer.com/wp-content/uploads/2020/02/Kalimba.mp3','https://www.youtube.com/watch?v=CmrG4Kvzf7I']

def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'text' in content_type.lower():
        return False
    if 'html' in content_type.lower():
        return False
    return True

def getFilename_fromCd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None

    fname = re.findall('filename=(.+)', cd)

    if len(fname) == 0:
        return None

    return fname[0]

# print(is_downloadable(url))


r = requests.get(url[0], allow_redirects=True)
filetype = r.headers.get('content-type')
print(filetype)


filename = getFilename_fromCd(r.headers.get('content-disposition'))

if filename == None:
    filename = "sinhala-song.mp3"

with open(filename, 'wb') as f:
    f.write(r.content)




