from pydub import AudioSegment
from os import listdir
from os.path import isfile, join
import csv

import_folder = "/home/neda.ahmadi/DutchSpeechRecognition/dutchdataset/clips"
export_folder = "/home/neda.ahmadi/DutchSpeechRecognition/clips_flac"
files = [f for f in listdir(import_folder) if isfile(join(import_folder, f))]


for file in files:
    # file = file[:-3]
    # print("in the for loop")
    try:
        song = AudioSegment.from_mp3(import_folder + "/" +file)
	
        song.export(export_folder+"/"+file[:-4]+".wav",format = "wav")

        # print(export_folder+"/"+file+"wav")
    except:
        print("continue")
        print(file)
        continue
