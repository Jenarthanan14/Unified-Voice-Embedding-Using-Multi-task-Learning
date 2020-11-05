import json
import logging
import os
from collections import deque, Counter
from random import choice
from time import time

import dill
import numpy as np
from tqdm import tqdm

from audio import pad_mfcc, Audio
from constants import NUM_FRAMES, NUM_FBANKS
from conv_models import DeepSpeakerModel
from utils import ensures_dir, load_pickle, load_npy, train_test_sp_to_utt

logger = logging.getLogger(__name__)

import pandas as pd 

def get_gender_label(speaker):
    result = pd.read_csv('/hive-unified-model/libri_speaker_gender.csv', delimiter=",")
    speakers=list(result['speaker'])
    gender_labels=list(result['gender'])
    gender=gender_labels[speakers.index(speaker)]
    if gender=='F':
        return 1
    else:
        return 0

def extract_speaker(utt_file):
    return utt_file.split('/')[-1].split('_')[0]


def sample_from_mfcc(mfcc, max_length):
    if mfcc.shape[0] >= max_length:
        r = choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r:r + max_length]
    else:
        s = pad_mfcc(mfcc, max_length)
    return np.expand_dims(s, axis=-1)


def sample_from_mfcc_file(utterance_file, max_length):
    mfcc = np.load(utterance_file)
    return sample_from_mfcc(mfcc, max_length)


class KerasFormatConverter:

    def __init__(self, working_dir, load_test_only=False):
        self.working_dir = working_dir
        self.output_dir = os.path.join(self.working_dir, 'keras-inputs')
        ensures_dir(self.output_dir)
        self.categorical_speakers = load_pickle(os.path.join(self.output_dir, 'categorical_speakers.pkl'))
        if not load_test_only:
            self.kx_train = load_npy(os.path.join(self.output_dir, 'kx_train.npy'))
            self.ky_train = load_npy(os.path.join(self.output_dir, 'ky_train.npy'))
            self.kg_train= load_npy(os.path.join(self.output_dir, 'kg_train.npy'))
        self.kx_test = load_npy(os.path.join(self.output_dir, 'kx_test.npy'))
        self.ky_test = load_npy(os.path.join(self.output_dir, 'ky_test.npy'))
        self.kg_test = load_npy(os.path.join(self.output_dir, 'kg_test.npy'))
        self.audio = Audio(cache_dir=self.working_dir, audio_dir=None)
        if self.categorical_speakers is None:
            self.categorical_speakers = SparseCategoricalSpeakers(self.audio.speaker_ids)

    def persist_to_disk(self):
        with open(os.path.join(self.output_dir, 'categorical_speakers.pkl'), 'wb') as w:
            dill.dump(self.categorical_speakers, w)
        np.save(os.path.join(self.output_dir, 'kx_train.npy'), self.kx_train)
        np.save(os.path.join(self.output_dir, 'kx_test.npy'), self.kx_test)
        np.save(os.path.join(self.output_dir, 'ky_train.npy'), self.ky_train)
        np.save(os.path.join(self.output_dir, 'kg_train.npy'), self.kg_train)
        np.save(os.path.join(self.output_dir, 'ky_test.npy'), self.ky_test)
        np.save(os.path.join(self.output_dir, 'kg_test.npy'), self.kg_test)

    def generate_per_phase(self, max_length=NUM_FRAMES, num_per_speaker=3000, is_test=False):
        # train OR test.
        num_speakers = len(self.audio.speaker_ids)
        sp_to_utt = train_test_sp_to_utt(self.audio, is_test)

        # 64 fbanks 1 channel(s).
        # float32
        kx = np.zeros((num_speakers * num_per_speaker, max_length, NUM_FBANKS, 1), dtype=np.float32)
        ky = np.zeros((num_speakers * num_per_speaker, 1), dtype=np.float32)
        kg=np.zeros((num_speakers * num_per_speaker, 1), dtype=np.float32)

        desc = f'Converting to Keras format [{"test" if is_test else "train"}]'
        for i, speaker_id in enumerate(tqdm(self.audio.speaker_ids, desc=desc)):
            utterances_files = sp_to_utt[speaker_id]
            for j, utterance_file in enumerate(np.random.choice(utterances_files, size=num_per_speaker, replace=True)):
                self.load_into_mat(utterance_file, self.categorical_speakers, speaker_id, max_length, kx, ky,kg,
                                   i * num_per_speaker + j)
        return kx, ky,kg

    def generate(self, max_length=NUM_FRAMES, counts_per_speaker=(3000, 500)):
        kx_train, ky_train,kg_train = self.generate_per_phase(max_length, counts_per_speaker[0], is_test=False)
        kx_test, ky_test,kg_test = self.generate_per_phase(max_length, counts_per_speaker[1], is_test=True)
        logger.info(f'kx_train.shape = {kx_train.shape}')
        logger.info(f'ky_train.shape = {ky_train.shape}')
        logger.info(f'kg_train.shape = {kg_train.shape}')
        logger.info(f'kx_test.shape = {kx_test.shape}')
        logger.info(f'ky_test.shape = {ky_test.shape}')
        logger.info(f'kg_test.shape = {kg_test.shape}')
        self.kx_train, self.ky_train, self.kg_train,self.kx_test, self.ky_test,self.kg_test = kx_train, ky_train,kg_train, kx_test, ky_test,kg_test

    @staticmethod
    def load_into_mat(utterance_file, categorical_speakers, speaker_id, max_length, kx, ky,kg, i):
        kx[i] = sample_from_mfcc_file(utterance_file, max_length)
        ky[i] = categorical_speakers.get_index(speaker_id)
        kg[i]= get_gender_label(int(speaker_id))

class SparseCategoricalSpeakers:

    def __init__(self, speakers_list):
        self.speaker_ids = sorted(speakers_list)
        assert len(set(self.speaker_ids)) == len(self.speaker_ids)  # all unique.
        self.map = dict(zip(self.speaker_ids, range(len(self.speaker_ids))))

    def get_index(self, speaker_id):
        return self.map[speaker_id]
