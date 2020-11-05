from os import walk
from glob import glob
import os
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from constants import CHECKPOINT_PATH

f = []
speakers=[]

def find_files(directory, pattern='**/*.flac'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

f=find_files('.hive-mtl-model-wd/LibriSpeech/test-clean/')

for i in f:
	speakers.append(i.split('/')[-1].split('-')[0])

print(f[0])
speaker_no=[]
speaker_u=list(set(speakers))
print(speaker_u)
test_speaker =[]
enroll_speaker=[]
speakerID_enroll=[]
speakerID_test=[]
speaker_u.sort()
print(len(speaker_u))


import random

import numpy as np

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity
from utils import ensures_dir, load_pickle, load_npy, train_test_sp_to_utt

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
dsm = DeepSpeakerModel(include_softmax=False)
base_model = dsm.m
x = base_model.output
x = Dense(1024, name='shared')(x)
x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln1')(x)
model = Model(base_model.input,x)
# Load the checkpoint.
model.load_weights(CHECKPOINT_PATH, by_name=True)

for i in speaker_u:
	temp_speaker=[]
	for j in range(len(f)):
		if speakers[j]==i:
			temp_speaker.append(f[j])
	for k in range(len(temp_speaker)):
		if k==0:
			enroll_speaker.append(temp_speaker[k])
			speakerID_enroll.append(i)
		else:
			test_speaker.append(temp_speaker[k])
			speakerID_test.append(i)

count=0
for i in range(len(test_speaker)):
	mfcc_test=sample_from_mfcc(read_mfcc(test_speaker[i], SAMPLE_RATE), NUM_FRAMES)
	predict_002 =model.predict(np.expand_dims(mfcc_test, axis=0))
	print(predict_002.shape)
	max_score= -10**8
	pred_speaker=None
	true_speaker=speakerID_test[i]
	for j in range(len(enroll_speaker)):
		mfcc_enroll=sample_from_mfcc(read_mfcc(enroll_speaker[j], SAMPLE_RATE), NUM_FRAMES)
		predict_001 =model.predict(np.expand_dims(mfcc_enroll, axis=0))
		score=batch_cosine_similarity(predict_001, predict_002)
		if score>max_score:
			max_score=score
			pred_speaker=speakerID_enroll[j]
	print("True speaker : %s\nPredicted speaker : %s\nResult : %s\n" %(true_speaker, pred_speaker, true_speaker==pred_speaker))
	if pred_speaker==true_speaker:
		count+=1
print('accuracy: ',count/len(test_speaker))


