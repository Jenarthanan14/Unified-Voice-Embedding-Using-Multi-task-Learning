import logging
import os
import numpy as np 
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

from batcher import KerasFormatConverter, LazyTripletBatcher
from constants import BATCH_SIZE, CHECKPOINTS_SOFTMAX_DIR, NUM_FRAMES, NUM_FBANKS
from conv_models import DeepSpeakerModel
from triplet_loss import deep_speaker_loss
from utils import load_best_checkpoint, ensures_dir,train_test_sp_to_utt
import pandas as pd
logger = logging.getLogger(__name__)

# Otherwise it's just too much logging from Tensorflow...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def fit_model_mtl(dsm, kx_train, ky_train, kg_train,kx_test, ky_test, kg_test,
                      batch_size=BATCH_SIZE, max_epochs=1000, initial_epoch=0):
    checkpoint_name =  'multitask'+ '_checkpoint'
    checkpoint_filename = os.path.join(CHECKPOINTS_MTL_DIR, checkpoint_name + '_{epoch}.h5')

    tb = TensorBoard(log_dir=os.path.join(CHECKPOINTS_MTL_DIR,'logs', 'HiveMtlModel'))

    checkpoint = ModelCheckpoint(monitor='val_speaker_pred_accuracy', filepath=checkpoint_filename, save_best_only=True)

    # if the accuracy does not increase by 0.1% over 20 epochs, we stop the training.
    early_stopping = EarlyStopping(monitor='val_speaker_pred_accuracy', min_delta=0.001, patience=20, verbose=1, mode='max')

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_speaker_pred_accuracy', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    max_len_train = len(kx_train) - len(kx_train) % batch_size
    kx_train = kx_train[0:max_len_train]
    ky_train = ky_train[0:max_len_train]
    (unique,counts)=np.unique(ky_train,return_counts=True)
    frequencies=np.asarray((unique,counts)).T
    kg_train=kg_train[0:max_len_train]
    max_len_test = len(kx_test) - len(kx_test) % batch_size
    kx_test = kx_test[0:max_len_test]
    ky_test = ky_test[0:max_len_test]
    kg_test = kg_test[0:max_len_test]

    dsm.fit(x=kx_train,
              y=[ky_train,kg_train],
              batch_size=batch_size,
              epochs=initial_epoch + max_epochs,
              initial_epoch=initial_epoch,
              verbose=1,
              shuffle=True,
              validation_data=(kx_test, [ky_test,kg_test]),
              callbacks=[tb,early_stopping, reduce_lr, checkpoint,CustomCallback()])


def start_training(working_dir):
    pre_training_phase=True
    ensures_dir(CHECKPOINTS_SOFTMAX_DIR)
    ensures_dir(CHECKPOINTS_TRIPLET_DIR)
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    logger.info('Started training.')
    kc = KerasFormatConverter(working_dir)
 
    num_speakers_softmax = len(kc.categorical_speakers.speaker_ids)
    logger.info(f'categorical_speakers: {kc.categorical_speakers.speaker_ids}')
    dsm = DeepSpeakerModel(batch_input_shape, include_softmax=False, num_speakers_softmax=num_speakers_softmax)
    base_model = dsm.m
    x = base_model.output
    x = Dense(1024, name='shared')(x)
    y=Dense(1024,name='speaker_task')(x)
    speaker_out= Dense(num_speakers_softmax, activation='softmax',name='speaker_pred')(y)
    gender_out= Dense(1, activation='sigmoid',name='gender_pred')(x)
    model = Model(inputs=base_model.input, outputs=[speaker_out, gender_out])
    
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy','binary_crossentropy'], metrics={'speaker_pred': 'accuracy', 'gender_pred': 'binary_accuracy'})
    training_checkpoint = load_best_checkpoint(CHECKPOINTS_MTL_DIR)
    if training_checkpoint is not None:
        initial_epoch = int(training_checkpoint.split('/')[-1].split('.')[0].split('_')[-1])
        logger.info(f'Initial epoch is {initial_epoch}.')
        logger.info(f'Loading softmax checkpoint: {training_checkpoint}.')
        model.load_weights(training_checkpoint)  # latest one.
    else:
        initial_epoch = 0
    fit_model_mtl(model, kc.kx_train, kc.ky_train,kc.kg_train, kc.kx_test, kc.ky_test,kc.kg_test, initial_epoch=initial_epoch)
  
