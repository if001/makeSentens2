import numpy as np
import matplotlib.pylab as plt
from keras.models import Model

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Sequential
# from keras.layers.wrappers import TD

from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda, Input, Dense, GRU, LSTM, RepeatVector, concatenate, Dropout, Bidirectional
from keras.models import Model
from keras.layers.core import Flatten
from keras.layers import merge, multiply
from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.normalization import BatchNormalization as BN

from keras import regularizers

from . import nn

import sys
import os


class Seq2SeqConfig():
    def __init__(self):
        self.word_feat_len = 128
        self.latent_dim = 512
        self.latent_dim2 = 1024
        self.batch_size = 10
        self.loss = 'mean_squared_error'
        self.optimizer = 'rmsprop'

        #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        #optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True)
        # optimizer = 'Adam'


class Seq2Seq():
    def __init__(self, flag="train"):
        self.nn = nn.NN("./models/weight/test.hdf5")
        self.cnf = Seq2SeqConfig()

        if flag == "train":
            self.__make_net()
        elif flag == "make":
            self.sequence_autoencoder = self.nn.load_models()
            self.__make_decode_net()
        elif flag == "resume":
            self.sequence_autoencoder = self.nn.load_models()
        else:
            print("invalid flag!!")
            exit(0)
        self.model_complie()

    def __make_net(self):
        input_dim = self.cnf.word_feat_len
        output_dim = self.cnf.word_feat_len

        encoder_inputs = Input(shape=(None, input_dim))
        encoder_dense_outputs = Dense(
            input_dim, activation='sigmoid')(encoder_inputs)
        encoder_bi_lstm = LSTM(
            self.cnf.latent_dim, return_sequences=True, dropout=0.6, recurrent_dropout=0.6)
        encoder_bi_outputs = Bi(encoder_bi_lstm)(encoder_dense_outputs)
        _, state_h, state_c = LSTM(self.cnf.latent_dim, return_state=True,
                                   dropout=0.2, recurrent_dropout=0.2)(encoder_bi_outputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, input_dim))
        decoder_dense_outputs = Dense(
            input_dim, activation='sigmoid')(decoder_inputs)
        decoder_bi_lstm = LSTM(
            self.cnf.latent_dim, return_sequences=True, dropout=0.6, recurrent_dropout=0.6)
        decoder_bi_outputs = Bi(decoder_bi_lstm)(decoder_dense_outputs)
        decoder_lstm = LSTM(self.cnf.latent_dim, return_sequences=True,
                            return_state=True, dropout=0.2, recurrent_dropout=0.2)
        decoder_outputs, _, _ = decoder_lstm(
            decoder_bi_outputs, initial_state=encoder_states)
        decoder_outputs = Dense(output_dim, activation='relu')(decoder_outputs)
        decoder_outputs = Dense(
            output_dim, activation='linear')(decoder_outputs)

        self.sequence_autoencoder = Model(
            [encoder_inputs, decoder_inputs], decoder_outputs)

    def __make_decode_net(self):
        input_dim = self.cnf.word_feat_len
        output_dim = self.cnf.word_feat_len

        ei, di, ed, dd, eb, db, el, dl, dd2, dd3 = self.sequence_autoencoder.layers

        encoder_inputs = Input(shape=(None, input_dim))
        encoder_dense_output = Dense(
            input_dim, activation='sigmoid', weights=ed.get_weights())(encoder_inputs)
        encoder_bi_output = eb(encoder_dense_output)
        _, state_h, state_c = LSTM(
            self.cnf.latent_dim, return_state=True, weights=el.get_weights())(encoder_bi_output)

        decoder_states_inputs = [state_h, state_c]

        decoder_inputs = Input(shape=(None, input_dim))
        decoder_dense_outputs = Dense(
            input_dim, activation='sigmoid', weights=dd.get_weights())(decoder_inputs)
        decoder_lstm_outputs = db(decoder_dense_outputs)
        decoder_lstm = LSTM(self.cnf.latent_dim, return_sequences=True,
                            return_state=True, weights=dl.get_weights())
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(
            decoder_lstm_outputs, initial_state=decoder_states_inputs)
        decoder_outputs = Dense(
            output_dim, activation='relu', weights=dd2.get_weights())(decoder_outputs)
        decoder_outputs = Dense(
            output_dim, activation='linear', weights=dd3.get_weights())(decoder_outputs)

        self.decoder_model = Model(
            [encoder_inputs, decoder_inputs], [decoder_outputs, decoder_state_h, decoder_state_c])

    def model_complie(self):
        """ complie """
        self.sequence_autoencoder.compile(optimizer=self.cnf.optimizer,
                                          loss=self.cnf.loss,
                                          metrics=['accuracy'])
        self.sequence_autoencoder.summary()

    def train(self, encoder_input_data, decoder_input_data, decoder_target_data):
        """ Run training """
        loss = self.sequence_autoencoder.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                             batch_size=self.cnf.batch_size,
                                             epochs=1)

        return loss

    def make_sentens_vec(self, input_sentens_arr, start_token):
        sentens_vec = []
        word_vec = start_token
        end_len = 20
        stop_condition = False

        while not stop_condition:
            word_vec, h, c = self.decoder_model.predict(
                [input_sentens_arr, word_vec])
            sentens_vec.append(word_vec)
            states_value = [h, c]
            if (sentens_vec == 0 or len(sentens_vec) == end_len):
                stop_condition = True
        return sentens_vec

    def save_model(self):
        self.nn.save_models(self.sequence_autoencoder)
