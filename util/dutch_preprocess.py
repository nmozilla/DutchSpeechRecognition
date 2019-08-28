from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.io.wavfile as wav
from python_speech_features import logfbank

import argparse
import csv
from os import listdir
from os.path import isfile, join


#dev_path = '/home/neda.ahmadi/GermanSpeechRecognition/dutchdataset/dev.tsv'
#train_path = '/home/neda.ahmadi/GermanSpeechRecognition/dutchdataset/train.tsv'
#test_path = '/home/neda.ahmadi/GermanSpeechRecognition/dutchdataset/test.tsv'
#root = '/home/neda.ahmadi/GermanSpeechRecognition/clips_flac/'

dev_path = '/data/s3559734/DutchDS/dev.tsv'
train_path = '/data/s3559734/DutchDS/train.tsv'
test_path = '/data/s3559734/DutchDS/test.tsv'
root = '/data/s3559734/DutchSpeechRecognition/clips_flac/'

n_jobs = -2
n_filters = 40
win_size = 0.025/3
norm_x = False

def traverse(root,path,search_fix='.wav',return_label=False):
 files = sorted(os.listdir(root))
 numfiles = len(files)
 if path == "train":
  set = files[:int(0.7*numfiles)]
 elif path == "dev":
  set = files[int(0.7*numfiles):int(0.9 *numfiles)]
 else:
  set = files[int(0.9*numfiles):]
 f_list = []
 with open('/data/s3559734/DutchDS/validated.tsv') as txt_file:
  reader = csv.reader(txt_file, delimiter='\t')
 # print('reader',reader)
  for line in reader:
   if (line[1][:-4] + ".wav") in set:
     if return_label:
      f_list.append(line[2])
     else:
      f_list.append(root + line[1][:-4]+".wav")
 return f_list

def wav2logfbank(f_path):
    (rate,sig) = wav.read(f_path)
    fbank_feat = logfbank(sig,rate,winlen=win_size,nfilt=n_filters)
    np.save(f_path[:-3]+'fb'+str(n_filters),fbank_feat)


def norm(f_path,mean,std):
    np.save(f_path,(np.load(f_path)-mean)/std)


print('----------Processing Datasets----------')
print('Training sets :',train_path)
print('Validation sets :',dev_path)
print('Testing sets :',test_path)

tr_file_list = traverse(root,"train")
dev_file_list = traverse(root,"dev")
tt_file_list = traverse(root,"test")


# # wav 2 log-mel fbank
print('---------------------------------------')
print('Processing wav2logfbank...',flush=True)

print('Training',flush=True)
results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2logfbank)(i[:-4]+'.wav') for i in tqdm(tr_file_list))

print('Validation',flush=True)
results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2logfbank)(i[:-4]+'.wav') for i in tqdm(dev_file_list))

print('Testing',flush=True)
results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2logfbank)(i[:-4]+'.wav') for i in tqdm(tt_file_list))
                    
# # log-mel fbank 2 feature
print('---------------------------------------') 
print('Preparing Training Dataset...',flush=True)

tr_file_list = traverse(root,"train",search_fix='.fb'+str(n_filters))
tr_text = traverse(root,"train",return_label=True)

X = []
for f in tr_file_list:
   # print(np.load(f[:-3]+"fb40.npy"))
   # X.append(np.load(f))
   X.append(np.load(f[:-3] +"fb40.npy"))    

# Normalize X
if norm_x:
    mean_x = np.mean(np.concatenate(X,axis=0),axis=0)
    std_x = np.std(np.concatenate(X,axis=0),axis=0)

    results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(norm)(i,mean_x,std_x) for i in tqdm(tr_file_list))

# Sort data by signal length (long to short)
audio_len = [len(x) for x in X]

tr_file_list = [tr_file_list[idx] for idx in reversed(np.argsort(audio_len))]
tr_text = [tr_text[idx] for idx in reversed(np.argsort(audio_len))]

# Create char mapping
char_map = {}
char_map['<sos>'] = 0
char_map['<eos>'] = 1
char_idx = 2

# map char to index
for text in tr_text:
    for char in text:
        if char not in char_map:
            char_map[char] = char_idx
            char_idx +=1

# Reverse mapping
rev_char_map = {v:k for k,v in char_map.items()}

# Save mapping
with open(root+'idx2chap.csv','w') as f:
    f.write('idx,char\n')
    for i in range(len(rev_char_map)):
        f.write(str(i)+','+rev_char_map[i]+'\n')

# text to index sequence
tmp_list = []
for text in tr_text:
    tmp = []
    for char in text:
        tmp.append(char_map[char])
    tmp_list.append(tmp)
tr_text = tmp_list
del tmp_list

# write dataset
file_name = 'train.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)

with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(tr_file_list)):
        f.write(str(i)+',')
        f.write(tr_file_list[i]+',')
        for char in tr_text[i]:
            f.write(' '+str(char))
        f.write('\n')

print()
print('Preparing Validation Dataset...',flush=True)

dev_file_list = traverse(root,"dev",search_fix='.fb'+str(n_filters))
dev_text = traverse(root,"dev",return_label=True)

X = []
for f in dev_file_list:
    X.append(np.load(f[:-3] +"fb40.npy"))
# Normalize X
if norm_x:
    results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(norm)(i,mean_x,std_x) for i in tqdm(dev_file_list))
# Sort data by signal length (long to short)
audio_len = [len(x) for x in X]

dev_file_list = [dev_file_list[idx] for idx in reversed(np.argsort(audio_len))]
dev_text = [dev_text[idx] for idx in reversed(np.argsort(audio_len))]

# text to index sequence
tmp_list = []
for text in dev_text:
    tmp = []
    for char in text:
        tmp.append(char_map[char])
    tmp_list.append(tmp)
dev_text = tmp_list
del tmp_list

# write dataset
file_name = 'dev.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)
with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(dev_file_list)):
        f.write(str(i)+',')
        f.write(dev_file_list[i]+',')
        for char in dev_text[i]:
            f.write(' '+str(char))
        f.write('\n')

print()
print('Preparing Testing Dataset...',flush=True)

test_file_list = traverse(root,"test",search_fix='.fb'+str(n_filters))
tt_text = traverse(root,"test",return_label=True)

X = []
for f in test_file_list:
    X.append(np.load(f[:-3] +"fb40.npy"))

# Normalize X
if norm_x:
    results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(norm)(i,mean_x,std_x) for i in tqdm(test_file_list))

# Sort data by signal length (long to short)
audio_len = [len(x) for x in X]

test_file_list = [test_file_list[idx] for idx in reversed(np.argsort(audio_len))]
tt_text = [tt_text[idx] for idx in reversed(np.argsort(audio_len))]

# text to index sequence
tmp_list = []
for text in tt_text:
    tmp = []
    for char in text:
        try:
            tmp.append(char_map[char])
        except:
            print(char)
    tmp_list.append(tmp)
tt_text = tmp_list
del tmp_list

# write dataset
file_name = 'test.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)

with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(test_file_list)):
        f.write(str(i)+',')
        f.write(test_file_list[i]+',')
        for char in tt_text[i]:
            f.write(' '+str(char))
        f.write('\n')
