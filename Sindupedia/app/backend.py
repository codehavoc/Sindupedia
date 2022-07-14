# Imports
import os
import requests
import validators
from datetime import timedelta
from flask import Flask, current_app, redirect, url_for, render_template, request, session, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, \
     check_password_hash
from werkzeug.utils import secure_filename

# ML 
import pandas as pd
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import librosa.display
import shutil
import youtube_dl
from collections import Counter
from PIL import Image, ImageDraw
import re

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Keras
import keras
import tensorflow as tf
from keras.models import model_from_json

app = Flask(__name__)
# ====================================== CUSTOM VARIABLES =================================================================================================
db_configs = {
    'user' : 'root',
    'password' : '',
    'host' : 'localhost',
    'database' : 'sindupedia'
}
db_song_ints = {
    'classics' : 0,
    'dance' : 1,
    'disco' : 2,
    'hiphop' : 3,
    'pop' : 4,
    'rnb' : 5,
    'soul' : 6,
    'swingandjazz' : 7,
}
genreDictionary = {
    'classics' : 'Classics',
    'dance' : 'Dance',
    'disco' : 'Disco',
    'hiphop' : 'Hip-Hop',
    'pop' : 'Pop',
    'rnb' : 'R&B',
    'soul' : 'Soul',
    'swingandjazz' : 'Swing & Jazz',
}
genreList = [
    'classics',
    'dance',
    'disco',
    'hiphop',
    'pop',
    'rnb',
    'soul',
    'swingandjazz',
]
genres = ['classics','dance','disco','hiphop']
numofgenres = len(genres)
imh = 120
imw = 270

np.random.seed(23456)
tf.random.set_seed(123)

model = None
scaler = StandardScaler()
# =========================================================================================================================================================
# ====================================== APP CONFIGS ======================================================================================================
app.secret_key = 'udaranayan2018158'
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql://{db_configs['user']}:{db_configs['password']}@{db_configs['host']}/{db_configs['database']}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(minutes=30)
app.debug = True
db = SQLAlchemy(app)
# =========================================================================================================================================================
# ====================================== DB MODELS ========================================================================================================
class users(db.Model):
    id = db.Column(db.Integer, primary_key = True, autoincrement = True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100),nullable=False)
    email = db.Column(db.String(100),nullable=False)
    password_encrypted = db.Column(db.String(200),nullable=False)

    def __init__(self,fname,lname,email,password):
        self.first_name = fname
        self.last_name = lname
        self.email = email
        self.password_encrypted = password

    def set_password(password):
        return generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_encrypted, password)

class songs(db.Model):
    id = db.Column(db.Integer, primary_key = True, autoincrement = True)
    original_name = db.Column(db.String(100), nullable=False)
    file_name = db.Column(db.String(400), nullable=False)
    genre = db.Column(db.Integer,nullable=False)
    path = db.Column(db.String(100),nullable=False)

    def __init__(self,oname,fname,genre,path):
        self.original_name = oname
        self.file_name = fname
        self.genre = genre
        self.path = path
    
    def generate_filename(filename):
        return secure_filename(filename)
# =========================================================================================================================================================
# =================================== USER DEFINED FUNCTIONS ==============================================================================================
class Waveform(object):
    bar_count = 107
    db_ceiling = 60

    def __init__(self, filename):
        self.filename = filename

        audio_file = AudioSegment.from_file(
            self.filename, self.filename.split('.')[-1])

        self.peaks = self._calculate_peaks(audio_file)

    def _calculate_peaks(self, audio_file):
        """ Returns a list of audio level peaks """
        chunk_length = len(audio_file) / self.bar_count

        loudness_of_chunks = [
            audio_file[i * chunk_length: (i + 1) * chunk_length].rms
            for i in range(self.bar_count)]

        max_rms = max(loudness_of_chunks) * 1.00

        return [int((loudness / max_rms) * self.db_ceiling)
                for loudness in loudness_of_chunks]

    def _get_bar_image(self, size, fill):
        """ Returns an image of a bar. """
        width, height = size
        bar = Image.new('RGBA', size, fill)

        end = Image.new('RGBA', (width, 2), fill)
        draw = ImageDraw.Draw(end)
        draw.point([(0, 0), (3, 0)], fill='#c1c1c1')
        draw.point([(0, 1), (3, 1), (1, 0), (2, 0)], fill='#555555')

        bar.paste(end, (0, 0))
        bar.paste(end.rotate(180), (0, height - 2))
        return bar

    def _generate_waveform_image(self):
        """ Returns the full waveform image """
        im = Image.new('RGB', (840, 128), '#ffffff')
        for index, value in enumerate(self.peaks, start=0):
            column = index * 8 + 2
            upper_endpoint = 64 - value

            im.paste(self._get_bar_image((4, value * 2), '#000000'),
                     (column, upper_endpoint))

        return im

    def save(self,name):
        """ Save the waveform as an image """
        png_filename = name
        with open(png_filename, 'wb') as imfile:
            self._generate_waveform_image().save(imfile, 'PNG')



def melSpecGenerations(filename):
    imagename = re.sub(".wav", ".png", filename)
    imagename = re.sub("prediction","melspec",imagename)

    if os.path.exists(os.path.join(current_app.root_path, 'static/plotted_graphs', imagename)):
        print("[SERVER] :: Melspectrogram Already Exists...")
        return imagename

    else:
        print("[SERVER] :: Melspectrogram Not Found. Generating A New One...")
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots()

        y, sr = librosa.load(os.path.join(current_app.root_path, 'static/extracted_wavs', filename),duration=20, sr=22050) # your file

        S = librosa.feature.melspectrogram(y=y, sr=sr,  n_fft=2048, hop_length=1024)
        S = librosa.power_to_db(S, ref=np.max)
        S.resize(imh,imw, refcheck=False)

        librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)

        plt.savefig(os.path.join(current_app.root_path, 'static/plotted_graphs', imagename))

        return imagename



def authCheck(form,query):
    # print("came into check results ===========>")
    a = True
    validities = [True,True]

    if query:
        if form['email'] != query.email:
            a = False
            validities[0] = False
        if not users.check_password(query,form['password']):
            a = False
            validities[1] = False

        return [a,validities]
    else:
        flash("This account does not exists.","error")
        return [False,[True,True]]



def clearFolders():
    for f in os.listdir(os.path.join(current_app.root_path,'static/submissions')):
        os.remove(os.path.join(current_app.root_path, 'static/submissions', f))   

    for f in os.listdir(os.path.join(current_app.root_path,'static/extracted_wavs')):
        os.remove(os.path.join(current_app.root_path, 'static/extracted_wavs', f))

    for f in os.listdir(os.path.join(current_app.root_path,'static/plotted_graphs')):
        os.remove(os.path.join(current_app.root_path, 'static/plotted_graphs', f))



def getErrorMesseges(res):
    errorTexts = [
        f'Mail is incorrect.',
        f'Password is incorrect.'
    ]

    me = ''
    pe = f'We\'ll never share your password with anyone.'

    if res[0] == False:
        me = errorTexts[0]
    if res[1] == False:
        pe = errorTexts[1]
    return [me,pe]



def createTestSongs():
    resourcePath = 'static\songs'
    for dir in  os.listdir(resourcePath):
        for f in os.listdir(f'{resourcePath}\{dir}'):
            new_name = f
            newSong = songs(
                    oname = f,
                    fname = new_name,
                    genre = db_song_ints[dir],
                    path = resourcePath
                )
            db.session.add(newSong)
    db.session.commit()    



def createTestUsers():
    userList = [
        {'fname':'Udara','lname':'Nayana','email':'udaranayana.111@gmail.com','password':'1234'},
        {'fname':'Test','lname':'User','email':'test@gmail.com','password':'1234'}
    ]

    for user in userList:
        hashed_password = users.set_password(user['password'])
        newUser = users(
            fname = user['fname'],
            lname = user['lname'],
            email = user['email'],
            password = hashed_password
        )
        db.session.add(newUser)

    db.session.commit()   



def addSongToDb(ofilename,filename,maingenre):
    check = songs.query.filter(songs.original_name == filename).all()

    if len(check) == 0:
        resourcePath = 'static\songs'
        addNewSong = songs(
                oname = ofilename,
                fname = filename,
                genre = maingenre,
                path = resourcePath
        )
        db.session.add(addNewSong)
        db.session.commit()



def getFilenamefromCd(cd):
    if not cd:
        return None

    fname = re.findall('filename=(.+)', cd)

    if len(fname) == 0:
        return None

    return fname[0]



def downloadTheYoutubeVideo(video_url):
    try:
        video_info = youtube_dl.YoutubeDL().extract_info(
            url = video_url,download=False
        )
        originalname = f"{video_info['title']}.mp3"
        filename = songs.generate_filename(originalname)
        options={
            'format':'bestaudio/best',
            'keepvideo':False,
            'outtmpl': os.path.join(current_app.root_path, 'static\submissions', filename),
        }

        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([video_info['webpage_url']])

        return True,originalname,filename
    except:
        return False,'',''



def validYoutubeLink(link):
    if 'www.youtube.com/watch?' in link:
        return True
    else:
        return False



def if_downloadable(url):

    valid=validators.url(url)

    if not valid:
        return False,'','',''
    
    print("[SERVER] :: Provided URL was valid..")

    # If a youtube link
    if validYoutubeLink(url):
        print(f"[SERVER] :: Provided link is a youtube link...")
        state,originalname,filename = downloadTheYoutubeVideo(url)

        return state,originalname,filename,'webm'

    # If not a Youtube link
    else:
        try:
            h = requests.head(url, allow_redirects=True)

            header = h.headers
            content_type = header.get('content-type')
            print(f"[SERVER] :: Content type ({content_type})")

            if 'text' in content_type.lower():
                return False,'','',''
            if 'html' in content_type.lower():
                return False,'','',''
            
            r = requests.get(url, allow_redirects=True)

            print("[SERVER] :: File was downloadable...")

            originalname = getFilenamefromCd(r.headers.get('content-disposition'))
            if originalname == None:
                search = "%{}%".format('sinhala-songs')
                count = songs.query.filter(songs.original_name.like(search)).count()
                originalname = f"sinhala-songs-{str(count+1).zfill(2)}.mp3"
                filename = songs.generate_filename(originalname)

            with open(os.path.join(current_app.root_path, 'static\submissions', filename), 'wb') as f:
                f.write(r.content)

            return True,originalname,filename,'mp3'
        except:
            return False,'','',''



def modelLoading():
    global model
    print("[SERVER] :: Start Loading Model From The Disk...")

    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("model/model.h5")

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    print("[SERVER] :: Model Loaded From The Disk Successfully...")



def audioClipping(song,filetype):
    try:
        type = "prediction"
        savepath = os.path.join(current_app.root_path,'static/extracted_wavs')
        genre = f"{type}."
        count = 0

        timestamps = []
        timestampLength = 20

        print("[SERVER] :: Starting Audio Clipping...")
        
        duration = int(librosa.get_duration(filename=song))

        while timestampLength < duration:
            timestamps.append(timestampLength)
            timestampLength += 20

        newAudio = AudioSegment.from_file(song,filetype)

        for x in range(0,len(timestamps)-1):

            t1 = timestamps[x]*1000
            t2 = timestamps[x+1]*1000
            
            clippedWav = newAudio[t1:t2]
            wavcount = f"{genre}{str(count).zfill(4)}.wav"

            # print("saving ",wavcount)

            clippedWav.export(f"{savepath}/{wavcount}", format="wav")
            count = count + 1

        print("[SERVER] :: Ending Audio Clipping...")

        return True,timestamps
    except:
        return False,[] 



def featureExtractions():
    featureSets = []
    savepath = os.path.join(current_app.root_path,'static/extracted_wavs')
    
    print("[SERVER] :: Starting Feature Extractions...")

    for filename in os.listdir(savepath):
        songname = f'{savepath}/{filename}'
        duration = int(librosa.get_duration(filename=songname))

        if duration >= 17:
            y, sr = librosa.load(songname, mono=True, duration=20)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            
            featureSets.append(to_append.split())
        else:
            print(f'{filename} is defective')


    df = pd.DataFrame(featureSets)

    print("[SERVER] :: Ending Feature Extractions...")

    return df



def predictionFunction(df):
    global scaler
    global model

    predictionData = scaler.fit_transform(np.array(df, dtype = float))
    predictions = model.predict(predictionData)

    return predictions



def displayGenres(predictions):
    subgenres = []

    for i in range(0,len(predictions)):
        subgenres.append(np.argmax(predictions[i]))

    freCheck = [item for items, c in Counter(subgenres).most_common() for item in [items] * c]

    sortedGenres = []

    for i in freCheck:
        if i in sortedGenres:
            continue
        else:
            sortedGenres.append(i)

    return sortedGenres[0],sortedGenres,subgenres



def convertTime(value):
    seconds = value % ( 24 * 3600 )
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%02d:%02d" % (minutes,seconds)



def recreateDatabase():
    print("[SERVER] :: Dropping The Existing Database...")
    createTestSongs()
    createTestUsers()
    print("[SERVER] :: Database Reconstructed Successfully...")



def runCustomFunctions():
    print("[SERVER] :: Executing Custom Functions...")
    modelLoading()
    print("[SERVER] :: 200 : Custom Function Executed Successfully...")

    return True    
# =========================================================================================================================================================
# =========================================================================================================================================================
@app.route("/", methods=['POST','GET'])
def login():
    helptext = f'We\'ll never share your password with anyone else.'

    if request.method == "POST":
        check_user = users.query.filter_by(email=request.form['email']).first()
        # check_user = users.query.all()
        
        # print(check_user)

        results = authCheck(request.form,check_user)

        session.permanent = True
        if results[0]:
            session["auth"] = True

            authdata = {}
            authdata['fname'] = check_user.first_name
            authdata['lname'] = check_user.last_name
            authdata['email'] = check_user.email
            session["authdata"] = authdata
        else:
            session["auth"] = False
            session["authdata"] = {}

        messeges = getErrorMesseges(results[1])
            
        if session["auth"]: 
            session["showModal"] = False
            return redirect(url_for("homePage"))
        else:
            return render_template(
                "login.html",
                mailDetails = {'noerror': results[1][0],'messege': messeges[0]},
                passwordDetails = {'noerror': results[1][1],'messege': messeges[1]}
            )
    else:
        return render_template(
                "login.html",
                mailDetails = {'noerror': True,'messege':''},
                passwordDetails = {'noerror': True,'messege':helptext}
            )



@app.route("/Sign-in", methods=['POST','GET'])
def signIn():
    if request.method == "POST":
        session.permanent = True
        
        check_user = users.query.filter_by(email=request.form['email']).first()

        if check_user:
            flash("This email already exists in the system.",'error')
            return render_template("signin.html")
        else:    
            # print("request form is ======>",request.form)
            hashed_password = users.set_password(request.form['password'])
            # print("new hased password is ======>",hashed_password)
            newUser = users(
                    fname = request.form['fname'],
                    lname = request.form['lname'],
                    email = request.form['email'],
                    password = hashed_password
                )
            db.session.add(newUser)
            db.session.commit()
         
        session["auth"] = True

        userdata = users.query.filter_by(email=request.form['email']).first()

        authdata = {}
        authdata['fname'] = userdata.first_name
        authdata['lname'] = userdata.last_name
        authdata['email'] = userdata.email
        session["authdata"] = authdata

        session["showModal"] = True

        return redirect(url_for("homePage")) 

    else:
        session["auth"] = False
        session["authdata"] = {}
        return render_template("signin.html")



@app.route("/Log-out")
def logout():
    session.pop("auth",None)
    session.pop("authdata",None)
    session.pop("initials",None)

    session.pop('_flashes', None)
    flash("Logged out successfully.",'info')
    return redirect(url_for('login'))



@app.route("/Home")
def homePage():
    if "auth" in session and session['auth']:
        name = f'{session["authdata"]["fname"]} {session["authdata"]["lname"]}'
        namelist = name.split()
        initials = ''
        for i in namelist:
            initials += i[0]
        session['initials'] = initials

        showModal = False 
        if session["showModal"]:
            showModal = True
            session["showModal"] = False

        return render_template("index.html",modalState = showModal)
    else:
        session.pop('_flashes', None)
        flash("You are not logged in.",'error')
        return redirect(url_for('login'))



@app.route("/Home/Genres/<string:genre>")
@app.route("/Home/Genres/<string:genre>/<string:keyword>")
def directToGenre(genre,keyword = None):
    chunkSize = 4
    
    if keyword:
        search = "%{}%".format(keyword)
        results = songs.query.filter(songs.original_name.like(search),songs.genre == db_song_ints[genre]).all()
    else:
        results = songs.query.filter_by(genre = db_song_ints[genre]).all()

    chunkedData=[results[i:i + chunkSize] for i in range(0, len(results), chunkSize)]

    return render_template('genres.html',genreTile = genreDictionary[genre], songlist = chunkedData, genre = genre)



@app.route('/Extract-keywords',methods=['POST'])
def getKeywords():
    queryKeys = request.form['keyword']
    gnre = request.form['gnre']

    return redirect(url_for('directToGenre',genre = gnre, keyword = queryKeys))



@app.route('/Extract-keywords/All-genres',methods=['POST'])
def getKeywordsForAllGens():
    queryKeys = request.form['keyword']

    return redirect(url_for('directToAllGenres',keyword = queryKeys))



@app.route("/Home/All-genres")
@app.route("/Home/All-genres/<string:keyword>")
def directToAllGenres(keyword = None):
    chunkSize = 4

    if keyword:
        search = "%{}%".format(keyword)
        results = songs.query.filter(songs.original_name.like(search)).all()
    else:
        results = songs.query.order_by(songs.genre.asc()).all()

    chunkedData=[results[i:i + chunkSize] for i in range(0, len(results), chunkSize)]

    return render_template('allgenres.html', songlist = chunkedData)



@app.route("/Home/Prediction-Results",methods = ['POST'])
def collectFormData():
    run = False
    msg = ''
    originalname = ''
    filename = ''
    filetype = ''
    showDownloadBtn = ''

    clearFolders()

    file = request.files['file']
    link = request.form.get('urls')

    if link:
        result_state,originalname,filename,filetype = if_downloadable(link)
        showDownloadBtn = filename
        run = result_state
        
        if result_state == False:
            msg = "The link seems to be invalid or incomplete. We couldn't download your song !!"

    elif file.filename != '':
        filename = songs.generate_filename(file.filename)
        file.save(os.path.join(current_app.root_path,'static/submissions',filename))
        a, extension = os.path.splitext(os.path.join(current_app.root_path,'static/submissions',filename))

        if extension == '.mp3':
            originalname = file.filename
            filetype = 'mp3'
            run = True
        else:
            msg = "Please make sure the file type is a .mp3 !!"

    if run:
        excutionState,timestamps = audioClipping(os.path.join(current_app.root_path,'static/submissions',filename),filetype)

        # make sure no errors happend during this
        if not excutionState:
            msg = "Sorry, The file was corrupted !!"
            flash(msg,"error")
            return redirect(url_for('homePage')) 

        # feature extraction and predictions
        dataframe = featureExtractions()
        presults = predictionFunction(dataframe)
        maingenre,subgenres,genrePerClip = displayGenres(presults)

        numOfClips = len(timestamps)-1
        newTimestamp = []
        wavform_names = []
        allAudioClips = os.listdir(os.path.join(current_app.root_path,'static/extracted_wavs'))

        # prevent image generation errors from custom class or matlab
        try:
            for file in allAudioClips:
                newName = re.sub('.wav','.png',file)

                newFile = Waveform(os.path.join(current_app.root_path,'static/extracted_wavs',file))
                newFile.save(os.path.join(current_app.root_path,'static/plotted_graphs',newName))

                wavform_names.append(newName)
        except:
            msg = "Sorry, The file was corrupted !!"
            flash(msg,"error")
            return redirect(url_for('homePage'))

        # create timestamps strings
        for i in timestamps:
            newTime = convertTime(i)
            newTimestamp.append(str(newTime))

        # copy file to related genre collection
        shutil.copy2(os.path.join(current_app.root_path,'static/submissions',filename), os.path.join(current_app.root_path,'static/songs',genreList[maingenre],filename))
        
        #create new db record for the added song
        addSongToDb(originalname,filename,maingenre)

        return render_template(
            'analysis.html',
            maingenre = maingenre,
            subgenre = subgenres,
            clipGenre = genrePerClip,
            timestamps = newTimestamp,
            allFiles = allAudioClips,
            wavforms = wavform_names,
            noc = numOfClips,
            sdb = showDownloadBtn,
            songName = filename
        )
    else:
        if msg == '':
            msg = "Please provide audio source materials !!"
        
        flash(msg,"error")
        return redirect(url_for('homePage'))   



@app.route("/Download/<int:id>")
def downloads(id):
    song = songs.query.filter_by(id=id).first()

    strorage_folder = f'static/songs/{genreList[song.genre]}'
    path = os.path.join(current_app.root_path, strorage_folder)

    return send_from_directory(path,song.file_name)



@app.route('/View-Audio-Clip/<string:name>')
def directToTheAudioClip(name):
    strorage_folder = f'static/extracted_wavs'
    path = os.path.join(current_app.root_path, strorage_folder)

    return send_from_directory(path,name)



@app.route('/View-Spectrogram/<string:name>')
def directToThemelSpectrogram(name):
    strorage_folder = f'static/plotted_graphs'
    path = os.path.join(current_app.root_path, strorage_folder)

    img_name = melSpecGenerations(name)

    return send_from_directory(path,img_name)



@app.route('/Download/Youtube-Audio/<string:filename>')
def downloadGeneratedMp3(filename):
    strorage_folder = f'static/submissions'
    path = os.path.join(current_app.root_path, strorage_folder)
    
    return send_from_directory(path,filename)

# =========================================================================================================================================================
if __name__ == "__main__":
    # db.drop_all()
    # db.create_all()

    # Only excute when dropping the database
    # recreateDatabase()
    
    if runCustomFunctions():
        app.run()