import pandas as pd
import os
from tqdm import tqdm
from glob import glob
import youtube_dl

import os
import librosa
from pydub import AudioSegment

def trim_audio_data(audio_file, save_file, start):
    sr = 44100

    y, sr = librosa.load(audio_file, sr=sr)
    print("Save!")
    ny = y[sr*start:sr*(start+10)]
    librosa.write_wav(save_file + '.wav', ny, sr)

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '320',
    }],
}

# vggsound.csv : https://www.robots.ox.ac.uk/~vgg/data/vggsound/
vgg = pd.read_csv("vggsound.csv", names=["YouTube ID", "start seconds", "label", "train/test split"])

slink = "https://www.youtube.com/watch?v="

sumofError = 0
cnt = 0

os.makedirs("./vggsound", exist_ok=True)

for idx, row in tqdm(enumerate(vgg.iterrows())):
    try:
        _, row = row 
        url, sttime, label, split = row["YouTube ID"], row["start seconds"], row["label"], row["train/test split"] 
        endtime = int(sttime) + 10 
            
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([slink + url])

        # Save 10 sec Wav File with Text Prompt
        path = glob("*.mp3")[0]
        sound = AudioSegment.from_mp3(path)
        sound = sound[int(sttime) * 1000:int(endtime) * 1000]
        sound.export("./vggsound/"+label+str("_")+str(idx)+".wav", format="wav")
        os.remove(path)

    except:
        sumofError += 1
        continue
    
print(sumofError , "The number of error cases")